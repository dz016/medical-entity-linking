"""
train.py
--------
Trains a Word2Vec model on the MedMentions corpus and saves the result.

Hyperparameters are chosen for biomedical text:
  - vector_size=200   : richer than the classic 100-d; captures nuance in
                        medical terminology without over-parameterising on
                        ~4 k abstracts
  - window=6          : medical phrases can be multi-word; slightly wider
                        window than the standard 5
  - min_count=3       : keep low-frequency medical terms (rare ≠ unimportant)
  - sg=1              : Skip-Gram — better for infrequent words than CBOW
  - negative=10       : more negatives improves quality on domain-specific vocab
  - epochs=15         : small corpus → more passes
  - workers=4         : parallelism; tune to your machine

Usage:
    # full pipeline from scratch
    python train.py

    # point at an already-parsed corpus
    python train.py --corpus data/corpus_pubtator.txt --out models/

    # tweak hyperparams
    python train.py --vector-size 300 --window 8 --epochs 20
"""

import argparse
import logging
import time
from pathlib import Path

from gensim.models import Word2Vec

from download import download
from parse import iter_sentences

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy sentence iterator — streams from disk, never loads everything to RAM
# ---------------------------------------------------------------------------
class SentenceCorpus:
    """Re-iterable wrapper around iter_sentences so gensim can do multi-epoch."""

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path

    def __iter__(self):
        yield from iter_sentences(self.corpus_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    corpus_path: Path,
    out_dir: Path,
    vector_size: int = 200,
    window: int = 6,
    min_count: int = 3,
    epochs: int = 15,
    workers: int = 4,
) -> Word2Vec:
    out_dir.mkdir(parents=True, exist_ok=True)
    sentences = SentenceCorpus(corpus_path)

    log.info("Building vocabulary …")
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,           # Skip-Gram
        negative=10,
        workers=workers,
        seed=42,
    )
    model.build_vocab(sentences)
    log.info(
        "Vocabulary: %d unique tokens (min_count=%d)",
        len(model.wv),
        min_count,
    )

    log.info("Training Word2Vec (epochs=%d) …", epochs)
    t0 = time.time()
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
    )
    elapsed = time.time() - t0
    log.info("Training complete in %.1f s", elapsed)

    # Save in two formats:
    #   1. Native gensim format — reload with Word2Vec.load() for further training
    #   2. Word2Vec binary format — for inference/loading with KeyedVectors
    model_dir = out_dir / "word2vec_MedMentions"
    weights_dir = model_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_path = weights_dir / "word2vec_MedMentions.model"
    bin_path   = weights_dir / "word2vec_MedMentions.bin"
    kv_path    = weights_dir / "word2vec_MedMentions.kv"

    model.save(str(model_path))
    model.wv.save(str(kv_path))
    model.wv.save_word2vec_format(str(bin_path), binary=True)

    log.info("Model saved  → %s", model_path)
    log.info("Vectors saved→ %s", kv_path)
    log.info("Binary saved → %s", bin_path)
    

    return model


# ---------------------------------------------------------------------------
# Quick sanity-check after training
# ---------------------------------------------------------------------------
def sanity_check(model: Word2Vec) -> None:
    probes = [
        # (query_term, expected_neighbours)
        ("diabetes",      ["insulin", "glucose", "hyperglycemia"]),
        ("cancer",        ["tumor", "malignant", "carcinoma"]),
        ("inflammation",  ["cytokine", "inflammatory", "immune"]),
    ]

    print("\n── Sanity check: nearest neighbours ──")
    for word, hints in probes:
        if word not in model.wv:
            print(f"  '{word}' not in vocabulary — skipping")
            continue
        neighbours = [w for w, _ in model.wv.most_similar(word, topn=5)]
        overlap = [h for h in hints if h in neighbours]
        print(f"  {word:20s} → {neighbours}")
        if overlap:
            print(f"  {'':20s}   ✓ found expected: {overlap}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Word2Vec on MedMentions")
    parser.add_argument("--corpus", default=None,
                        help="Path to corpus_pubtator.txt. "
                             "If omitted, corpus is downloaded first.")
    parser.add_argument("--data-dir",    default="data",   help="Download dir")
    parser.add_argument("--out",         default="models", help="Model output dir")
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--window",      type=int, default=6)
    parser.add_argument("--min-count",   type=int, default=3)
    parser.add_argument("--epochs",      type=int, default=15)
    parser.add_argument("--workers",     type=int, default=4)
    args = parser.parse_args()

    corpus_path = Path(args.corpus) if args.corpus else download(Path(args.data_dir))

    model = train(
        corpus_path=corpus_path,
        out_dir=Path(args.out),
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
    )

    sanity_check(model)