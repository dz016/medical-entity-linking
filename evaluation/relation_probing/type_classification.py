from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _multi_hot(rows: list[dict], type_to_index: dict[str, int]) -> tuple[list[str], np.ndarray]:
    texts = []
    labels = []
    for row in rows:
        text = row["text"]
        vector = np.zeros(len(type_to_index), dtype=np.float32)
        for sty in row["types"]:
            if sty in type_to_index:
                vector[type_to_index[sty]] = 1.0
        if vector.sum() > 0:
            texts.append(text)
            labels.append(vector)
    return texts, np.vstack(labels) if labels else np.zeros((0, len(type_to_index)), dtype=np.float32)


def _macro_and_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, type_vocab: list[str]) -> tuple[float, dict[str, float]]:
    per_class = {}
    f1s = []
    for idx, name in enumerate(type_vocab):
        truth = y_true[:, idx]
        pred = y_pred[:, idx]
        tp = float(((truth == 1) & (pred == 1)).sum())
        fp = float(((truth == 0) & (pred == 1)).sum())
        fn = float(((truth == 1) & (pred == 0)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[name] = f1
        f1s.append(f1)
    return float(sum(f1s) / max(len(f1s), 1)), per_class


class TypeProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def evaluate_type_classification(
    embedder,
    cui_to_type_path: str,
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    max_examples: int | None = None,
) -> dict:
    payload = json.loads(Path(cui_to_type_path).read_text(encoding="utf-8"))
    type_vocab = payload["type_vocab"]
    type_to_index = payload["type_to_index"]
    rows = []
    for cui, types in payload["cui_to_types"].items():
        for text in payload["cui_to_terms"].get(cui, [])[:1]:
            rows.append({"text": text, "types": types})
    texts, labels = _multi_hot(rows, type_to_index)
    if max_examples is not None and max_examples > 0 and len(texts) > max_examples:
        texts = texts[:max_examples]
        labels = labels[:max_examples]
    if len(texts) < 10:
        return {"macro_f1": 0.0, "per_class_f1": {}, "examples": len(texts)}

    split = max(int(0.8 * len(texts)), 1)
    train_texts, test_texts = texts[:split], texts[split:]
    y_train, y_test = labels[:split], labels[split:]
    x_train = embedder.encode(train_texts, batch_size=batch_size)
    x_test = embedder.encode(test_texts, batch_size=batch_size)

    model = TypeProbe(x_train.shape[1], y_train.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
    )
    for _ in range(epochs):
        for features, targets in loader:
            optimiser.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimiser.step()

    with torch.inference_mode():
        logits = model(torch.from_numpy(x_test.astype(np.float32)))
        preds = (torch.sigmoid(logits).numpy() >= 0.5).astype(np.float32)
    macro_f1, per_class = _macro_and_per_class_f1(y_test, preds, type_vocab)
    return {"macro_f1": macro_f1, "per_class_f1": per_class, "examples": len(test_texts)}
