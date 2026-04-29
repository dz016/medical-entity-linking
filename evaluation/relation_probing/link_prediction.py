from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = [(s, y) for s, y in zip(scores, labels) if y == 1]
    negatives = [(s, y) for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0.0
    for pos, _ in positives:
        for neg, _ in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def evaluate_link_prediction(
    embedder,
    relation_pairs_path: str,
    relation_type: str = "has_manifestation",
    batch_size: int = 64,
    seed: int = 42,
    max_pairs: int | None = None,
) -> dict:
    rows = json.loads(Path(relation_pairs_path).read_text(encoding="utf-8"))
    positives = [row for row in rows if row["relation_type"] == relation_type]
    if max_pairs is not None and max_pairs > 0 and len(positives) > max_pairs:
        positives = positives[:max_pairs]
    if len(positives) < 2:
        return {"roc_auc": 0.0, "pairs": 0}
    candidates = sorted({row["positive_text"] for row in positives})
    rng = random.Random(seed)
    negatives = []
    for row in positives:
        wrong = rng.choice(candidates)
        if wrong == row["positive_text"] and len(candidates) > 1:
            wrong = candidates[(candidates.index(wrong) + 1) % len(candidates)]
        negatives.append((row["anchor_text"], wrong))

    pair_texts = [row["anchor_text"] for row in positives] + [row["positive_text"] for row in positives] + [neg[1] for neg in negatives]
    unique = list(dict.fromkeys(pair_texts))
    emb = embedder.encode(unique, batch_size=batch_size)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    lookup = {text: emb[idx] for idx, text in enumerate(unique)}

    labels, scores = [], []
    for row in positives:
        labels.append(1)
        scores.append(float(lookup[row["anchor_text"]] @ lookup[row["positive_text"]]))
    for anchor, wrong in negatives:
        labels.append(0)
        scores.append(float(lookup[anchor] @ lookup[wrong]))
    return {"roc_auc": _roc_auc(labels, scores), "pairs": len(labels)}
