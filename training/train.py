"""
training/train.py — single entry point for all training variants

Ablation variants (run from project root):

  Baseline — skip-gram / masked LM, no UMLS
  -----------------------------------------------
  python training/train.py --config training/configs/word2vec.json
  python training/train.py --config training/configs/transformer.json

  UMLS / Synonym Only — NT-Xent on synonym pairs only
  -----------------------------------------------
  python training/train.py --config training/configs/word2vec_umls.json
  python training/train.py --config training/configs/transformer_umls.json

  Synonym + Type — NT-Xent(syn) + lambda * BCE(types)
  -----------------------------------------------
  python training/train.py --config training/configs/word2vec_synonym_type.json
  python training/train.py --config training/configs/transformer_synonym_type.json

  Synonym + Relation — NT-Xent(syn) + NT-Xent(relations)
  -----------------------------------------------
  python training/train.py --config training/configs/word2vec_synonym_relation.json
  python training/train.py --config training/configs/transformer_synonym_relation.json

  Full Enhanced — NT-Xent(syn) + NT-Xent(rel) + lambda * BCE(types)
  -----------------------------------------------
  python training/train.py --config training/configs/word2vec_umls_enhanced.json
  python training/train.py --config training/configs/transformer_umls_enhanced.json

Before running alignment configs you must have a trained baseline model in models/.
Edit the UMLS file paths in each config to point at your local UMLS META/ directory.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.tasks import train_alignment_task, train_transformer_task, train_word2vec_task


def _infer_task(config: dict) -> str:
    if "alignment" in config:
        return "alignment"
    if "model" in config and "hidden_size" in config["model"]:
        return "transformer"
    return "word2vec"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON config file in training/configs/")
    args = parser.parse_args()

    config = load_config(args.config)
    task = _infer_task(config)

    print(f"\nrun   : {config.get('run_name', '?')}")
    print(f"task  : {task}\n")

    if task == "word2vec":
        train_word2vec_task(config)
    elif task == "transformer":
        train_transformer_task(config)
    else:
        train_alignment_task(config)


if __name__ == "__main__":
    main()
