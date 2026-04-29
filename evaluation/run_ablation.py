"""
evaluation/run_ablation.py — runs the 3 UMLS probing tasks for a given model

These measure how well the embedding space captures UMLS structure:
  - Link prediction (AUC)
  - Relational retrieval (MRR, P@20)
  - Semantic type classification (Macro F1)

Usage (run from project root):
  python evaluation/run_ablation.py --model word2vec_umls_enhanced \
      --relation_pairs training/data/relation_pairs.json \
      --cui_to_type training/data/cui_to_type.json

  # run for all models at once
  python evaluation/run_ablation.py --all \
      --relation_pairs training/data/relation_pairs.json \
      --cui_to_type training/data/cui_to_type.json

The --relation_pairs and --cui_to_type files are produced automatically
during alignment training (train_alignment_task writes them to training/data/).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))

from run_all import load_embedder
from relation_probing.link_prediction import evaluate_link_prediction
from relation_probing.relational_retrieval import evaluate_relational_retrieval
from relation_probing.type_classification import evaluate_type_classification


ALL_MODELS = [
    "word2vec",
    "word2vec_umls",
    "word2vec_umls_enhanced",
    "transformer_fast",
    "transformer_umls_fast",
    "transformer_umls_enhanced",
]


def run_probing(model_name: str, relation_pairs: str, cui_to_type: str, batch_size: int) -> dict:
    print(f"\n{'='*55}")
    print(f"  model: {model_name}")
    print(f"{'='*55}")

    embedder = load_embedder(model_name)

    print("\n[1/3] link prediction (AUC)...")
    lp = evaluate_link_prediction(
        embedder,
        relation_pairs_path=relation_pairs,
        relation_type="has_manifestation",
        batch_size=batch_size,
    )
    print(f"      AUC = {lp['roc_auc']:.4f}  ({lp['pairs']} pairs)")

    print("[2/3] relational retrieval (MRR, P@20)...")
    rr = evaluate_relational_retrieval(
        embedder,
        relation_pairs_path=relation_pairs,
        batch_size=batch_size,
    )
    print(f"      MRR = {rr['mrr']:.4f}   P@20 = {rr['precision@20']:.4f}  ({rr['queries']} queries)")

    print("[3/3] type classification (Macro F1)...")
    tc = evaluate_type_classification(
        embedder,
        cui_to_type_path=cui_to_type,
        batch_size=batch_size,
    )
    print(f"      Type F1 = {tc['macro_f1']:.4f}  ({tc['examples']} examples)")

    results = {
        "model": model_name,
        "roc_auc": lp["roc_auc"],
        "mrr": rr["mrr"],
        "precision_at_20": rr["precision@20"],
        "type_macro_f1": tc["macro_f1"],
    }

    out_dir = ROOT / "results" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_probing.json"
    with open(out_path, "w") as f:
        json.dump({**results, "link_prediction": lp, "relational_retrieval": rr, "type_classification": tc}, f, indent=2)
    print(f"\n  saved → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="single model name")
    group.add_argument("--all", action="store_true", help="run for all registered models")
    parser.add_argument("--relation_pairs", required=True,
                        help="path to relation_pairs.json (produced by training)")
    parser.add_argument("--cui_to_type", required=True,
                        help="path to cui_to_type.json (produced by training)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    models = ALL_MODELS if args.all else [args.model]
    rows = []
    for m in models:
        try:
            rows.append(run_probing(m, args.relation_pairs, args.cui_to_type, args.batch_size))
        except Exception as exc:
            print(f"  SKIPPED {m}: {exc}")

    if len(rows) > 1:
        print(f"\n{'='*55}")
        print(f"  {'Model':<30} {'AUC':>6} {'MRR':>6} {'P@20':>6} {'TypeF1':>7}")
        print(f"  {'-'*30} {'------':>6} {'------':>6} {'------':>6} {'-------':>7}")
        for r in rows:
            print(f"  {r['model']:<30} {r['roc_auc']:>6.4f} {r['mrr']:>6.4f} {r['precision_at_20']:>6.4f} {r['type_macro_f1']:>7.4f}")


if __name__ == "__main__":
    main()
