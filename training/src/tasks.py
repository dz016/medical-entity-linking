from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .alignment import train_alignment
from .backbones import (
    SentenceDataset,
    TransformerEncoderModel,
    Vocabulary,
    collate_masked_batch,
    train_word2vec_model,
)
from .export import export_alignment_model, export_transformer_baseline, export_word2vec_baseline
from .framework import RunContext
from .preprocessing import SimpleTokenizer, build_vocabulary, extract_umls_pairs, load_tokenized_corpus, materialize_corpus
from .utils import configure_logging, ensure_dir, set_seed


def ensure_common_artifacts(config: dict, require_umls_pairs: bool = False, keyed_vectors_path: str | None = None) -> None:
    data_cfg = config["data"]
    prep_cfg = config.get("preprocessing", {})
    tokenizer = SimpleTokenizer(min_token_length=prep_cfg.get("min_token_length", 2))

    tokenized_path = data_cfg.get("tokenized_corpus")
    if tokenized_path and not Path(tokenized_path).exists():
        materialize_corpus(
            pubmed_dir=data_cfg["pubmed_dir"],
            output_path=tokenized_path,
            tokenizer=tokenizer,
            max_sentences=data_cfg.get("max_sentences"),
        )

    vocab_path = data_cfg.get("vocab_json")
    if vocab_path and not Path(vocab_path).exists():
        build_vocabulary(
            corpus_path=tokenized_path,
            vocab_path=vocab_path,
            min_freq=prep_cfg.get("min_freq", 2),
            max_vocab_size=prep_cfg.get("max_vocab_size"),
        )

    if require_umls_pairs:
        pairs_path = data_cfg["pairs_txt"]
        if not Path(pairs_path).exists():
            extract_umls_pairs(
                mrconso_path=data_cfg["umls_mrconso"],
                output_pairs_path=pairs_path,
                language=prep_cfg.get("language", "ENG"),
                max_pairs_per_cui=prep_cfg.get("max_pairs_per_cui", 10),
                keyed_vectors_path=keyed_vectors_path,
            )


def train_word2vec_task(config: dict) -> None:
    configure_logging()
    set_seed(config["seed"])
    context = RunContext(config["run_name"], config["output"]["root"], config, resume=config["trainer"].get("resume", False))
    ensure_common_artifacts(config, require_umls_pairs=False)
    model = train_word2vec_model(load_tokenized_corpus(config["data"]["tokenized_corpus"]), config["model"], config["trainer"])
    export_word2vec_baseline(config, context, model.wv)


def train_transformer_task(config: dict) -> None:
    configure_logging()
    set_seed(config["seed"])
    context = RunContext(config["run_name"], config["output"]["root"], config, resume=config["trainer"].get("resume", False))
    ensure_common_artifacts(config, require_umls_pairs=False)

    vocab = Vocabulary.from_json(config["data"]["vocab_json"])
    dataset = SentenceDataset(
        corpus_path=config["data"]["tokenized_corpus"],
        vocab=vocab,
        max_length=config["model"]["max_length"],
        mask_probability=config["model"]["mask_probability"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_masked_batch(batch, vocab.pad_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoderModel(
        vocab_size=len(vocab.id_to_token),
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        ffn_dim=config["model"]["ffn_dim"],
        dropout=config["model"]["dropout"],
        max_length=config["model"]["max_length"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["trainer"]["lr"])

    start_epoch = 0
    if config["trainer"].get("resume"):
        checkpoint = context.load_checkpoint()
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, config["trainer"]["epochs"]):
        model.train()
        total_loss = 0.0
        for input_ids, labels, attention_mask in tqdm(loader, desc=f"transformer epoch {epoch+1}", unit="batch"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            _, logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        context.save_checkpoint({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()})

    export_transformer_baseline(config, context, model.cpu(), config["data"]["vocab_json"])


def train_alignment_task(config: dict) -> None:
    configure_logging()
    set_seed(config["seed"])
    context = RunContext(config["run_name"], config["output"]["root"], config, resume=config["trainer"].get("resume", False))
    keyed_vectors_path = None
    if config["alignment"]["base_model_type"] == "word2vec":
        keyed_vectors_path = str(Path(config["alignment"]["base_model_dir"]) / "weights" / "vectors.bin")
    if config.get("enhanced"):
        required = [
            config["data"].get("umls_mrconso"),
            config["data"].get("umls_mrsty"),
            config["data"].get("umls_mrrel"),
        ]
        missing = [path for path in required if not path or not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                "Enhanced alignment requires MRCONSO/MRSTY/MRREL files. "
                f"Missing: {', '.join(missing)}"
            )
    ensure_common_artifacts(config, require_umls_pairs=True, keyed_vectors_path=keyed_vectors_path)
    train_alignment(config, context, export_alignment_model)
