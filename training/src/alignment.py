import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .backbones import TransformerEncoderModel, Vocabulary
from .umls_enhanced import build_cui_to_type, build_relation_pairs


log = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    batch = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)
    logits = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(mask, float("-inf"))
    targets = torch.cat([torch.arange(batch, 2 * batch, device=z.device), torch.arange(0, batch, device=z.device)])
    return F.cross_entropy(logits, targets)


class UMLSPairDataset(Dataset):
    def __init__(self, pairs_path: str):
        self.pairs = []
        with Path(pairs_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    self.pairs.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


class EnhancedUMLSDataset(Dataset):
    def __init__(self, pairs_path: str, relation_pairs_path: str, relation_sampling_ratio: float, relation_types: list[str]):
        self.synonym_pairs = UMLSPairDataset(pairs_path).pairs
        payload = json_load(Path(relation_pairs_path))
        allowed = set(relation_types)
        self.relation_pairs = [row for row in payload if row["relation_type"] in allowed]
        self.relation_sampling_ratio = relation_sampling_ratio
        self.anchor_to_relations: dict[str, list[dict]] = defaultdict(list)
        for row in self.relation_pairs:
            self.anchor_to_relations[row["anchor_text"]].append(row)

    def __len__(self) -> int:
        return len(self.synonym_pairs)

    def __getitem__(self, idx: int):
        anchor, positive = self.synonym_pairs[idx]
        return {"anchor": anchor, "positive": positive}


class BackboneAdapter(nn.Module):
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError


class Word2VecBackboneAdapter(BackboneAdapter):
    def __init__(self, vectors_path: str, freeze: bool):
        super().__init__()
        kv = KeyedVectors.load_word2vec_format(vectors_path, binary=True)
        self.tokens = list(kv.key_to_index.keys())
        self.word_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(kv.vectors, dtype=torch.float32), freeze=freeze)

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        outputs = []
        device = self.embedding.weight.device
        for text in texts:
            ids = [self.word_to_id[token] for token in text.lower().split() if token in self.word_to_id]
            if not ids:
                outputs.append(torch.zeros(self.embedding_dim, device=device))
            else:
                tensor = torch.tensor(ids, dtype=torch.long, device=device)
                outputs.append(self.embedding(tensor).mean(dim=0))
        return torch.stack(outputs)

    def current_vectors(self) -> tuple[list[str], torch.Tensor]:
        return self.tokens, self.embedding.weight.detach().cpu()


class TransformerBackboneAdapter(BackboneAdapter):
    def __init__(self, model_dir: str, freeze: bool):
        super().__init__()
        metadata = json_load(Path(model_dir) / "metadata.json")
        vocab = Vocabulary.from_json(str(Path(model_dir) / "weights" / "vocab.json"))
        cfg = metadata["model_config"]
        self.pooling = metadata.get("pooling", "cls")
        self.vocab = vocab
        self.model = TransformerEncoderModel(
            vocab_size=len(vocab.id_to_token),
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"],
            dropout=cfg["dropout"],
            max_length=cfg["max_length"],
        )
        state = torch.load(Path(model_dir) / "weights" / "transformer.pt", map_location="cpu")
        self.model.load_state_dict(state)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self.model.token_embeddings.embedding_dim

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        sequences = [self.vocab.encode_tokens(text.lower().split(), max_length=self.model.position_embeddings.num_embeddings - 1, add_cls=True) for text in texts]
        max_len = max(len(seq) for seq in sequences)
        pad_id = self.vocab.pad_id
        input_ids = []
        masks = []
        for seq in sequences:
            pad = max_len - len(seq)
            input_ids.append(seq + [pad_id] * pad)
            masks.append([1] * len(seq) + [0] * pad)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        masks = torch.tensor(masks, dtype=torch.long, device=device)
        return self.model.encode(input_ids, masks, pooling=self.pooling)


def json_load(path: Path) -> dict:
    import json
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class AlignmentArtifacts:
    adapter: BackboneAdapter
    head: ProjectionHead
    type_classifier: nn.Module | None = None
    type_payload: dict | None = None


class TypeClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def load_alignment_components(base_model_type: str, base_model_dir: str, projection_dim: int, freeze_base: bool) -> AlignmentArtifacts:
    if base_model_type == "word2vec":
        adapter = Word2VecBackboneAdapter(str(Path(base_model_dir) / "weights" / "vectors.bin"), freeze=freeze_base)
    elif base_model_type == "transformer":
        adapter = TransformerBackboneAdapter(base_model_dir, freeze=freeze_base)
    else:
        raise ValueError(f"Unsupported base model type: {base_model_type}")
    head = ProjectionHead(adapter.embedding_dim, projection_dim)
    return AlignmentArtifacts(adapter=adapter, head=head)


def _text_to_types(type_payload: dict) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    cui_to_types = type_payload["cui_to_types"]
    cui_to_terms = type_payload["cui_to_terms"]
    for cui, types in cui_to_types.items():
        for text in cui_to_terms.get(cui, [])[:1]:
            mapping[text].update(types)
    return {text: sorted(types) for text, types in mapping.items()}


def _multi_hot(texts: list[str], text_to_types: dict[str, list[str]], type_to_index: dict[str, int], device: torch.device) -> torch.Tensor:
    output = torch.zeros((len(texts), len(type_to_index)), dtype=torch.float32, device=device)
    for row_idx, text in enumerate(texts):
        for sty in text_to_types.get(text, []):
            if sty in type_to_index:
                output[row_idx, type_to_index[sty]] = 1.0
    return output


def _build_class_weight_tensor(type_payload: dict, device: torch.device) -> torch.Tensor:
    weights = [float(type_payload["class_weights"][sty]) for sty in type_payload["type_vocab"]]
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights / weights.mean().clamp_min(1e-6)


def train_alignment(config: dict, context, export_fn) -> None:
    align_cfg = config["alignment"]
    trainer_cfg = config["trainer"]
    enhanced_cfg = config.get("enhanced")
    artifacts = load_alignment_components(
        base_model_type=align_cfg["base_model_type"],
        base_model_dir=align_cfg["base_model_dir"],
        projection_dim=align_cfg["projection_dim"],
        freeze_base=align_cfg["freeze_base"],
    )
    if enhanced_cfg:
        if align_cfg["base_model_type"] == "word2vec":
            vocab_source_path = str(Path(align_cfg["base_model_dir"]) / "weights" / "vectors.bin")
        else:
            vocab_source_path = str(Path(align_cfg["base_model_dir"]) / "weights" / "vocab.json")
        type_payload = build_cui_to_type(
            mrsty_path=config["data"]["umls_mrsty"],
            mrconso_path=config["data"]["umls_mrconso"],
            vocab_source_path=vocab_source_path,
            output_path=config["data"]["cui_to_type_json"],
            max_types=enhanced_cfg.get("max_types", 30),
        )
        build_relation_pairs(
            mrrel_path=config["data"]["umls_mrrel"],
            mrconso_path=config["data"]["umls_mrconso"],
            vocab_source_path=vocab_source_path,
            output_path=config["data"]["relation_pairs_json"],
            relation_types=enhanced_cfg.get("relation_types"),
        )
        dataset = EnhancedUMLSDataset(
            config["data"]["pairs_txt"],
            config["data"]["relation_pairs_json"],
            relation_sampling_ratio=enhanced_cfg.get("relation_sampling_ratio", 0.3),
            relation_types=enhanced_cfg.get("relation_types", []),
        )
        loader = DataLoader(dataset, batch_size=trainer_cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)
        artifacts.type_classifier = TypeClassifier(align_cfg["projection_dim"], len(type_payload["type_vocab"]))
        artifacts.type_payload = type_payload
    else:
        dataset = UMLSPairDataset(config["data"]["pairs_txt"])
        loader = DataLoader(dataset, batch_size=trainer_cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts.adapter = artifacts.adapter.to(device)
    artifacts.head = artifacts.head.to(device)
    if artifacts.type_classifier is not None:
        artifacts.type_classifier = artifacts.type_classifier.to(device)

    params = list(artifacts.head.parameters()) + [p for p in artifacts.adapter.parameters() if p.requires_grad]
    if artifacts.type_classifier is not None:
        params += list(artifacts.type_classifier.parameters())
    optimiser = torch.optim.AdamW(params, lr=trainer_cfg["lr"])

    start_epoch = 0
    if trainer_cfg.get("resume"):
        checkpoint = context.load_checkpoint()
        if checkpoint is not None:
            artifacts.head.load_state_dict(checkpoint["head"])
            artifacts.adapter.load_state_dict(checkpoint["adapter"], strict=False)
            if artifacts.type_classifier is not None and "type_classifier" in checkpoint and checkpoint["type_classifier"] is not None:
                artifacts.type_classifier.load_state_dict(checkpoint["type_classifier"])
            optimiser.load_state_dict(checkpoint["optim"])
            start_epoch = checkpoint["epoch"] + 1

    total_steps = max((config["trainer"]["epochs"] - start_epoch) * max(len(loader), 1), 1)
    global_step = 0
    text_to_types = _text_to_types(artifacts.type_payload) if artifacts.type_payload else {}
    type_weight = _build_class_weight_tensor(artifacts.type_payload, device) if artifacts.type_payload else None

    for epoch in range(start_epoch, trainer_cfg["epochs"]):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"alignment epoch {epoch+1}", unit="batch"):
            optimiser.zero_grad()
            if isinstance(batch, dict):
                anchors = list(batch["anchor"])
                positives = list(batch["positive"])
            else:
                anchors, positives = batch
                anchors, positives = list(anchors), list(positives)

            z_a = artifacts.head(artifacts.adapter.encode_texts(anchors))
            z_b = artifacts.head(artifacts.adapter.encode_texts(positives))
            loss = nt_xent(z_a, z_b, temperature=align_cfg["temperature"])

            if enhanced_cfg and artifacts.type_classifier is not None and artifacts.type_payload is not None:
                relation_pairs = getattr(dataset, "anchor_to_relations", {})
                relation_sampling_ratio = enhanced_cfg.get("relation_sampling_ratio", 0.3)
                sampled_relation_rows = []
                for anchor in anchors:
                    if torch.rand(1).item() <= relation_sampling_ratio and relation_pairs.get(anchor):
                        sampled_relation_rows.append(relation_pairs[anchor][int(torch.randint(len(relation_pairs[anchor]), (1,)).item())])
                if sampled_relation_rows:
                    rel_a = [row["anchor_text"] for row in sampled_relation_rows]
                    rel_b = [row["positive_text"] for row in sampled_relation_rows]
                    z_rel_a = artifacts.head(artifacts.adapter.encode_texts(rel_a))
                    z_rel_b = artifacts.head(artifacts.adapter.encode_texts(rel_b))
                    loss = loss + nt_xent(z_rel_a, z_rel_b, temperature=align_cfg["temperature"])

                targets = _multi_hot(anchors + positives, text_to_types, artifacts.type_payload["type_to_index"], device)
                projected = torch.cat([z_a, z_b], dim=0)
                logits = artifacts.type_classifier(projected)
                lambda_value = enhanced_cfg.get("type_loss_weight", 0.1)
                if enhanced_cfg.get("type_loss_warmup", False):
                    warmup_steps = max(int(0.2 * total_steps), 1)
                    lambda_value *= min(global_step / warmup_steps, 1.0)
                type_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    pos_weight=type_weight,
                )
                loss = loss + lambda_value * type_loss

            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())
            global_step += 1

        context.save_checkpoint(
            {
                "epoch": epoch,
                "head": artifacts.head.state_dict(),
                "adapter": artifacts.adapter.state_dict(),
                "type_classifier": artifacts.type_classifier.state_dict() if artifacts.type_classifier is not None else None,
                "optim": optimiser.state_dict(),
            }
        )
        log.info("epoch=%s loss=%.4f", epoch + 1, total_loss / max(len(loader), 1))

    export_fn(config=config, context=context, artifacts=artifacts)
