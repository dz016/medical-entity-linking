import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors, Word2Vec

from .utils import read_json, write_json


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    unk_id: int
    cls_id: int
    mask_id: int

    @classmethod
    def from_json(cls, path: str) -> "Vocabulary":
        payload = read_json(path)
        tokens = payload["special_tokens"] + payload["tokens"]
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(
            token_to_id=token_to_id,
            id_to_token=tokens,
            pad_id=token_to_id["[PAD]"],
            unk_id=token_to_id["[UNK]"],
            cls_id=token_to_id["[CLS]"],
            mask_id=token_to_id["[MASK]"],
        )

    def encode_tokens(self, tokens: list[str], max_length: int | None = None, add_cls: bool = False) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if add_cls:
            ids = [self.cls_id] + ids
        if max_length is not None:
            ids = ids[:max_length]
        return ids


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_path: str, vocab: Vocabulary, max_length: int, mask_probability: float):
        self.sentences = []
        self.vocab = vocab
        self.max_length = max_length
        self.mask_probability = mask_probability
        with Path(corpus_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                tokens = json.loads(line)
                self.sentences.append(vocab.encode_tokens(tokens, max_length=max_length - 1, add_cls=True))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        ids = self.sentences[idx]
        inputs = list(ids)
        labels = [-100] * len(inputs)
        for pos in range(1, len(inputs)):
            if np.random.rand() < self.mask_probability:
                labels[pos] = inputs[pos]
                inputs[pos] = self.vocab.mask_id
        return inputs, labels


def collate_masked_batch(batch, pad_id: int):
    max_len = max(len(item[0]) for item in batch)
    input_ids = []
    labels = []
    attn_mask = []
    for ids, labs in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [-100] * pad)
        attn_mask.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.long),
    )


class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, ffn_dim: int, dropout: float, max_length: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        encoded = self.encoder(hidden, src_key_padding_mask=(attention_mask == 0))
        logits = self.lm_head(encoded)
        return encoded, logits

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pooling: str = "cls"):
        encoded, _ = self.forward(input_ids, attention_mask)
        if pooling == "cls":
            return encoded[:, 0]
        masked = attention_mask.unsqueeze(-1)
        summed = (encoded * masked).sum(dim=1)
        denom = masked.sum(dim=1).clamp_min(1)
        return summed / denom


def train_word2vec_model(corpus_iterable, model_cfg: dict, trainer_cfg: dict) -> Word2Vec:
    return Word2Vec(
        sentences=corpus_iterable,
        vector_size=model_cfg["vector_size"],
        window=model_cfg["window"],
        min_count=model_cfg["min_count"],
        negative=model_cfg.get("negative", 10),
        sg=model_cfg.get("sg", 1),
        workers=trainer_cfg["workers"],
        epochs=trainer_cfg["epochs"],
        seed=42,
    )


def save_word2vec_export(root: str, kv: KeyedVectors, metadata: dict) -> None:
    weights = Path(root) / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    kv.save_word2vec_format(str(weights / "vectors.bin"), binary=True)
    write_json(Path(root) / "metadata.json", metadata)

