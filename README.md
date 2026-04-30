# Medical Entity Linking — Biomedical Embedding Benchmark

A systematic evaluation framework for biomedical text embeddings, benchmarking custom-trained models against pretrained baselines across three clinically relevant NLP tasks.

---

## Motivation

Embedding quality is the silent bottleneck in clinical NLP pipelines. A model that clusters *diabetes mellitus* near *insulin resistance* but far from *type 2 diabetes* will silently degrade every downstream task — entity linking, retrieval, and clinical decision support alike.

This project builds a reproducible evaluation harness to answer a concrete question: **can lightweight, domain-adapted models trained from scratch close the gap with large pretrained biomedical transformers?**

---

## Benchmark Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| Entity Linking — diseases | NCBI Disease | Acc@1, Acc@5, MRR |
| Entity Linking — diseases | BC5CDR-diseases | Acc@1, Acc@5, MRR |
| Entity Linking — chemicals | BC5CDR-chemicals | Acc@1, Acc@5, MRR |
| Semantic Textual Similarity | BIOSSES | Pearson r, Spearman r |
| Natural Language Inference | NLI4CT | Accuracy, Macro F1 |

Entity linking is performed against the full CTD knowledge base (~13k disease terms, ~10k chemical terms). Published numbers typically use a corpus-specific candidate KB (which yields higher Acc@1); our results reflect the harder full-KB setting.

---

## Results

### Entity Linking — NCBI Disease (Acc@1 / Acc@5 / MRR)

| Model | Acc@1 | Acc@5 | MRR |
|-------|-------|-------|-----|
| SapBERT (baseline) | **0.607** | **0.730** | **0.664** |
| Word2Vec + UMLS | 0.313 | 0.452 | 0.374 |
| Word2Vec + UMLS (enhanced) | 0.298 | 0.443 | 0.358 |
| Word2Vec | 0.284 | 0.416 | 0.341 |
| Transformer + UMLS (enhanced) | 0.302 | 0.445 | 0.367 |
| Transformer + UMLS | 0.160 | 0.195 | 0.172 |
| Transformer | 0.148 | 0.160 | 0.162 |

### Entity Linking — BC5CDR Diseases (Acc@1 / Acc@5 / MRR)

| Model | Acc@1 | Acc@5 | MRR |
|-------|-------|-------|-----|
| SapBERT (baseline) | **0.721** | **0.860** | **0.780** |
| Transformer + UMLS (enhanced) | 0.490 | 0.619 | 0.544 |
| Word2Vec + UMLS (enhanced) | 0.428 | 0.610 | 0.512 |
| Word2Vec | 0.463 | 0.627 | 0.532 |
| Word2Vec + UMLS | 0.425 | 0.631 | 0.520 |
| Transformer + UMLS | 0.369 | 0.396 | 0.381 |
| Transformer | 0.369 | 0.382 | 0.375 |

### Semantic Textual Similarity — BIOSSES (Pearson r / Spearman r)

| Model | Pearson r | Spearman r |
|-------|-----------|------------|
| SapBERT (baseline) | **0.883** | **0.875** |
| Word2Vec + UMLS (enhanced) | 0.706 | 0.735 |
| Word2Vec + UMLS | 0.640 | 0.627 |
| Word2Vec | 0.604 | 0.620 |
| Transformer + UMLS | 0.473 | 0.376 |
| Transformer + UMLS (enhanced) | 0.420 | 0.389 |
| Transformer | 0.192 | 0.179 |

### Natural Language Inference — NLI4CT (Accuracy / Macro F1)

| Model | Accuracy | Macro F1 | Majority Baseline |
|-------|----------|----------|-------------------|
| Transformer | **0.588** | **0.502** | 0.579 |
| SapBERT (baseline) | 0.584 | 0.534 | 0.579 |
| Word2Vec + UMLS | 0.571 | 0.523 | 0.579 |
| Transformer + UMLS (enhanced) | 0.575 | 0.389 | 0.579 |
| Word2Vec | 0.553 | 0.441 | 0.579 |
| Word2Vec + UMLS (enhanced) | 0.518 | 0.445 | 0.579 |
| Transformer + UMLS | 0.543 | 0.452 | 0.579 |

**Key finding:** Word2Vec with UMLS grounding reaches 71% of SapBERT's NCBI Acc@1 with a fraction of the compute and no transformer architecture. On BIOSSES, the enhanced Word2Vec variant achieves a Pearson r of 0.706 vs. SapBERT's 0.883 — a competitive result for a lightweight model. NLI is near-majority-baseline for all models, suggesting it requires fine-tuning rather than embedding-based inference.

---

## Project Structure

```
medical-entity-linking/
├── data/
│   ├── raw/              # datasets (not tracked in git — see setup below)
│   └── lookups/          # CTD KB files (diseases + chemicals)
├── models/               # model implementations (weights not tracked in git)
│   ├── pubmedbert-local/ # pretrained baseline
│   ├── sapbert-local/    # pretrained baseline
│   ├── word2vec/         # custom Word2Vec model
│   ├── word2vec_umls/    # Word2Vec grounded on UMLS
│   ├── transformer/      # custom transformer model
│   └── transformer_umls/ # transformer grounded on UMLS
├── evaluation/
│   ├── base_embedder.py          # abstract interface all models implement
│   ├── pubmedbert_embedder.py    # HuggingFace transformer wrapper
│   ├── eval_entity_linking.py    # NCBI Disease + BC5CDR evaluation
│   ├── eval_sts.py               # BIOSSES evaluation
│   ├── eval_nli.py               # NLI4CT evaluation
│   └── run_all.py                # orchestrates all tasks for a given model
├── notebooks/            # exploratory data analysis per dataset
├── results/              # benchmark outputs — JSON, figures, leaderboard CSV
├── CONTRIBUTING_MODELS.md
└── requirements.txt
```

---

## Design

### Uniform model interface

Every model — pretrained baselines and custom models alike — implements the same two-method interface:

```python
class BaseEmbedder(ABC):
    def load(self, model_path: str) -> None: ...
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...
```

Evaluation scripts never import model internals. Adding a new model requires only creating a subclass and registering the weights path — the benchmark runs unchanged.

### Evaluation pipeline

Entity linking uses nearest-neighbour search over the full CTD KB using cosine similarity. Semantic similarity uses Pearson and Spearman correlation against human-annotated sentence pairs. NLI uses embedding-based classification over clinical trial report pairs.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/your-org/medical-entity-linking.git
cd medical-entity-linking

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Download model weights from the shared Google Drive and place them in the corresponding `models/*/weights/` directories. See [CONTRIBUTING_MODELS.md](CONTRIBUTING_MODELS.md) for the expected structure.

---

## Running the Benchmark

Run all tasks for a single model:

```bash
cd evaluation
python run_all.py --model sapbert
```

Run a specific task:

```bash
python run_all.py --model word2vec_umls --task entity_linking
python run_all.py --model word2vec_umls --task sts
python run_all.py --model word2vec_umls --task nli
```

Run on a specific dataset:

```bash
python eval_entity_linking.py --model word2vec_umls --dataset ncbi
python eval_entity_linking.py --model word2vec_umls --dataset bc5cdr_d
python eval_entity_linking.py --model word2vec_umls --dataset bc5cdr_c
```

Results are written to `results/<model_name>/` as JSON files and figures. A combined leaderboard is regenerated at `results/leaderboard.csv` after each run.

---

## Adding a New Model

See [CONTRIBUTING_MODELS.md](CONTRIBUTING_MODELS.md) for the full guide — folder structure, required files, Word2Vec and Transformer worked examples, and a PR checklist.

The short version: subclass `BaseEmbedder`, implement `load()` and `encode()`, and register your model path in `run_all.py`.

```python
from evaluation.base_embedder import BaseEmbedder

class MyModel(BaseEmbedder):
    def load(self, model_path: str) -> None:
        # load weights from model_path
        ...

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        # return shape (N, embedding_dim), dtype float32
        ...
```

---

## Datasets

| Dataset | Task | Source |
|---------|------|--------|
| NCBI Disease | Entity linking | [Dogan et al., 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3861373/) |
| BC5CDR | Entity linking (diseases + chemicals) | [Li et al., 2016](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414) |
| BIOSSES | Semantic similarity | [Soğancıoğlu et al., 2017](https://academic.oup.com/bioinformatics/article/33/14/i49/3953954) |
| NLI4CT | Natural language inference | [Jullien et al., 2023](https://aclanthology.org/2023.semeval-1.188/) |
| CTD | Knowledge base for entity linking | [Davis et al., 2023](https://academic.oup.com/nar/article/51/D1/D1282/6830671) |
