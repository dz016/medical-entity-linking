"""
evaluation/eval_nli.py

Evaluates any BaseEmbedder on Natural Language Inference:
  - NLI4CT (Clinical Trial NLI)

How it works:
  1. encode(sentence1) -> vector A  (clinical trial context)
  2. encode(sentence2) -> vector B  (hypothesis)
  3. clean NaN/inf in embeddings (important for word2vec models)
  4. build feature vector: [A, B, A-B, A*B]
  5. train MLP classifier on train split
  6. evaluate on test split -> Accuracy + per-class F1

Why MLP over Logistic Regression?
  - feature vector is high dimensional (1200d for word2vec, 3072d for transformers)
  - MLP learns non-linear boundaries — important for NLI
  - early_stopping prevents overfitting on small data

Usage:
    python eval_nli.py --model pubmedbert --dataset nli4ct
    python eval_nli.py --model sapbert    --dataset nli4ct
    python eval_nli.py --model word2vec   --dataset nli4ct
"""

import json
import argparse
import time
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ─── paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
NLI4CT_DIR  = DATA_DIR / 'raw' / 'nli4ct'


# ─── dataset loader ───────────────────────────────────────────────────────────

def load_nli4ct(split: str = 'train'):
    """
    Load NLI4CT split.
    Returns (DataFrame, actual_split_name)
    DataFrame columns: sentence1, sentence2, label
    """
    file_map = {
        'train'     : NLI4CT_DIR / 'train.jsonl',
        'validation': NLI4CT_DIR / 'validation.jsonl',
        'test'      : NLI4CT_DIR / 'test.jsonl',
    }

    path = file_map.get(split)
    if not path or not path.exists():
        available = [s for s, p in file_map.items() if p.exists()]
        print(f'split "{split}" not found. using: {available[0]}')
        path  = file_map[available[0]]
        split = available[0]

    rows = []
    with open(path) as f:
        for line in f:
            row   = json.loads(line)
            label = row.get('gold_label') or row.get('label', '')
            rows.append({
                'sentence1': row['sentence1'].strip(),
                'sentence2': row['sentence2'].strip(),
                'label'    : label.strip()
            })

    df = pd.DataFrame(rows)
    print(f'NLI4CT {split}: {len(df)} examples loaded')
    print(f'label distribution:\n{df["label"].value_counts().to_string()}')
    return df, split


# ─── helpers ──────────────────────────────────────────────────────────────────

def clean_embeddings(emb: np.ndarray) -> np.ndarray:
    """
    Replace NaN and inf with 0.
    Critical for word2vec models where OOV inputs produce zero vectors
    and unnormalized vectors can cause overflow.
    """
    return np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Build NLI feature vector from two embeddings.

    Args:
        emb_a : (N, dim) embeddings of sentence1
        emb_b : (N, dim) embeddings of sentence2

    Returns:
        (N, 4*dim) feature matrix [A, B, A-B, A*B]
    """
    diff    = emb_a - emb_b
    product = emb_a * emb_b
    return np.hstack([emb_a, emb_b, diff, product])


# ─── main evaluation function ─────────────────────────────────────────────────

def evaluate(
    embedder,
    dataset     : str  = 'nli4ct',
    batch_size  : int  = 32,
    save_figures: bool = True,
    max_iter    : int  = 200
):
    """
    Run NLI evaluation for one model.

    Since NLI4CT only has a train split, we do an 80/20 internal split.

    Args:
        embedder    : any BaseEmbedder instance (already loaded)
        dataset     : 'nli4ct'
        batch_size  : passed to embedder.encode()
        save_figures: save plots to results/
        max_iter    : MLP max iterations
    """

    if dataset != 'nli4ct':
        raise ValueError(f'unknown dataset: {dataset}. choose nli4ct')

    # ── load data ──
    df, actual_split = load_nli4ct('train')

    # 80/20 split — stratified to keep class balance
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f'\ntrain: {len(train_df)}  test: {len(test_df)}')

    # ── encode sentences ──
    # encode() handles everything internally:
    # word2vec   → mean of word vectors (300d)
    # transformer → CLS or mean pool   (768d)
    # output always (N, dim)
    print(f'\nencoding train sentences...')
    t0 = time.time()

    train_emb_a = embedder.encode(train_df['sentence1'].tolist(), batch_size=batch_size)
    train_emb_b = embedder.encode(train_df['sentence2'].tolist(), batch_size=batch_size)

    print(f'encoding test sentences...')
    test_emb_a  = embedder.encode(test_df['sentence1'].tolist(), batch_size=batch_size)
    test_emb_b  = embedder.encode(test_df['sentence2'].tolist(), batch_size=batch_size)

    encode_time = time.time() - t0
    print(f'encoded in {encode_time:.1f}s — embedding dim: {train_emb_a.shape[1]}')

    # ── clean NaN and inf ──
    # critical for word2vec models — OOV terms produce zero vectors
    # and unnormalized vectors can cause overflow during feature building
    train_emb_a = clean_embeddings(train_emb_a)
    train_emb_b = clean_embeddings(train_emb_b)
    test_emb_a  = clean_embeddings(test_emb_a)
    test_emb_b  = clean_embeddings(test_emb_b)

    # ── build features ──
    # [A, B, A-B, A*B] → (N, 4*dim)
    X_train = build_features(train_emb_a, train_emb_b)
    X_test  = build_features(test_emb_a,  test_emb_b)

    # clean features too — product can still produce inf/NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ── encode labels ──
    le      = LabelEncoder()
    y_train = le.fit_transform(train_df['label'])
    y_test  = le.transform(test_df['label'])

    print(f'\nfeature shape : {X_train.shape}')
    print(f'classes       : {le.classes_}')

    # ── train MLP classifier ──
    # MLP handles high dimensional features better than logistic regression
    # StandardScaler normalizes features before MLP for faster convergence
    # early_stopping monitors validation loss to prevent overfitting
    print(f'\ntraining MLP classifier...')
    t1 = time.time()

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(512, 128),
            activation='relu',
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        ))
    ])

    clf.fit(X_train, y_train)
    train_time = time.time() - t1
    print(f'trained in {train_time:.1f}s')

    # ── evaluate ──
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    # majority baseline — always predicting most common class
    majority_label = train_df['label'].value_counts().index[0]
    majority_acc   = (test_df['label'] == majority_label).mean()

    total_time = encode_time + train_time

    result = {
        'model'            : embedder.name,
        'dataset'          : dataset,
        'classifier'       : 'MLP(512,128)',
        'accuracy'         : round(float(accuracy), 4),
        'majority_baseline': round(float(majority_acc), 4),
        'improvement'      : round(float(accuracy - majority_acc), 4),
        'per_class_f1'     : {
            cls: round(report[cls]['f1-score'], 4)
            for cls in le.classes_
        },
        'macro_f1'         : round(report['macro avg']['f1-score'], 4),
        'num_train'        : len(train_df),
        'num_test'         : len(test_df),
        'embedding_dim'    : int(train_emb_a.shape[1]),
        'feature_dim'      : int(X_train.shape[1]),
        'runtime_sec'      : round(total_time, 2),
        'date'             : str(date.today()),
    }

    print(f'\n{"="*50}')
    print(f'  model            : {result["model"]}')
    print(f'  dataset          : {result["dataset"]}')
    print(f'  classifier       : {result["classifier"]}')
    print(f'  accuracy         : {result["accuracy"]:.4f}')
    print(f'  majority baseline: {result["majority_baseline"]:.4f}')
    print(f'  improvement      : {result["improvement"]:+.4f}')
    print(f'  macro F1         : {result["macro_f1"]:.4f}')
    print(f'  per class F1:')
    for cls, f1 in result['per_class_f1'].items():
        print(f'    {cls:<20}: {f1:.4f}')
    print(f'  embedding dim    : {result["embedding_dim"]}')
    print(f'  feature dim      : {result["feature_dim"]}')
    print(f'  runtime          : {result["runtime_sec"]}s')
    print(f'{"="*50}\n')

    # ── save results ──
    out_dir = RESULTS_DIR / embedder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / f'nli_{dataset}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'results saved to {result_path}')

    # ── save figures ──
    if save_figures:
        fig_dir = out_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[0], cmap='Blues'
        )
        axes[0].set_title('Confusion matrix')
        axes[0].set_ylabel('true label')
        axes[0].set_xlabel('predicted label')

        # per class F1 bar chart
        classes = list(result['per_class_f1'].keys())
        f1s     = list(result['per_class_f1'].values())
        axes[1].bar(classes, f1s)
        axes[1].axhline(
            y=result['macro_f1'], color='r',
            linestyle='--', label=f'macro F1={result["macro_f1"]:.3f}'
        )
        axes[1].set_title('Per-class F1 score')
        axes[1].set_ylabel('F1')
        axes[1].set_ylim(0, 1)
        axes[1].legend()

        plt.suptitle(
            f'{embedder.name} | {dataset} | acc={accuracy:.4f}',
            fontsize=13
        )
        plt.tight_layout()

        fig_path = fig_dir / f'{dataset}_results.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f'figure saved to {fig_path}')

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True,
                        help='pubmedbert | sapbert | biobert | minilm | word2vec | word2vec_umls | transformer | transformer_umls')
    parser.add_argument('--dataset',    default='nli4ct',
                        help='nli4ct (default: nli4ct)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_iter',   type=int, default=200)
    args = parser.parse_args()

    ROOT_CLI = Path(__file__).parent.parent
    sys.path.insert(0, str(Path(__file__).parent))

    # ── load embedder ──
    if args.model == 'pubmedbert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT_CLI / 'models' / 'pubmedbert-local'))

    elif args.model == 'sapbert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT_CLI / 'models' / 'sapbert-local'))
        embedder._name = 'sapbert'

    elif args.model == 'biobert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT_CLI / 'models' / 'biobert-local'))
        embedder._name = 'biobert'

    elif args.model == 'minilm':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT_CLI / 'models' / 'minilm-local'))
        embedder._name = 'minilm'

    elif args.model == 'word2vec':
        sys.path.insert(0, str(ROOT_CLI / 'models' / 'word2vec'))
        from model import Word2VecEmbedder
        embedder = Word2VecEmbedder()
        embedder.load(str(ROOT_CLI / 'models' / 'word2vec'))

    elif args.model == 'word2vec_umls':
        sys.path.insert(0, str(ROOT_CLI / 'models' / 'word2vec_umls'))
        from model import Word2VecEmbedder
        embedder = Word2VecEmbedder()
        embedder.load(str(ROOT_CLI / 'models' / 'word2vec_umls'))

    else:
        raise ValueError(f'unknown model: {args.model}')

    evaluate(
        embedder   = embedder,
        dataset    = args.dataset,
        batch_size = args.batch_size,
        max_iter   = args.max_iter,
    )