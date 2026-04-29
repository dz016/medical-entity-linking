from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGET_RELATIONS = {
    "has_manifestation": "symptom",
    "may_treat": "drug",
    "treated_by": "drug",
    "finding_site_of": "anatomy",
}


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def evaluate_relational_retrieval(
    embedder,
    relation_pairs_path: str,
    batch_size: int = 64,
    max_queries: int | None = None,
) -> dict:
    rows = json.loads(Path(relation_pairs_path).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in rows:
        if row["relation_type"] in TARGET_RELATIONS:
            grouped[row["anchor_cui"]].append(row)
    anchors = sorted(grouped)
    if len(anchors) < 2:
        return {"precision@20": 0.0, "mrr": 0.0, "queries": 0}
    split = max(int(0.8 * len(anchors)), 1)
    test_anchors = set(anchors[split:])
    test_rows = [row for row in rows if row["anchor_cui"] in test_anchors and row["relation_type"] in TARGET_RELATIONS]
    if not test_rows:
        return {"precision@20": 0.0, "mrr": 0.0, "queries": 0}

    candidates = sorted({row["positive_text"] for row in test_rows})
    candidate_emb = _normalize(embedder.encode(candidates, batch_size=batch_size))
    candidate_index = {text: idx for idx, text in enumerate(candidates)}

    by_anchor = defaultdict(list)
    for row in test_rows:
        by_anchor[row["anchor_text"]].append(row["positive_text"])
    anchor_texts = sorted(by_anchor.keys())
    if max_queries is not None and max_queries > 0 and len(anchor_texts) > max_queries:
        anchor_texts = anchor_texts[:max_queries]
        by_anchor = {anchor: by_anchor[anchor] for anchor in anchor_texts}

    precisions = []
    rr = []
    anchor_embeddings = _normalize(embedder.encode(anchor_texts, batch_size=batch_size))
    for idx, anchor_text in enumerate(anchor_texts):
        positives = by_anchor[anchor_text]
        anchor_emb = anchor_embeddings[idx]
        scores = candidate_emb @ anchor_emb
        ranking = np.argsort(-scores)
        positive_set = {candidate_index[text] for text in positives if text in candidate_index}
        top20 = ranking[:20]
        precisions.append(sum(idx in positive_set for idx in top20) / 20.0)
        ranks = [int(np.where(ranking == idx)[0][0]) + 1 for idx in positive_set]
        rr.append(0.0 if not ranks else 1.0 / min(ranks))
    return {"precision@20": float(np.mean(precisions)), "mrr": float(np.mean(rr)), "queries": len(by_anchor)}
