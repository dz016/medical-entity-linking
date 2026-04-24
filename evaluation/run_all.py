"""
evaluation/run_all.py

Master script — runs all evaluations for a given model
across all 4 downstream tasks and builds leaderboard.csv

Usage:
    # run all tasks for pubmedbert
    python run_all.py --model pubmedbert

    # run specific task only
    python run_all.py --model pubmedbert --task entity_linking
    python run_all.py --model pubmedbert --task sts
    python run_all.py --model pubmedbert --task nli

    # run for team model (once they push)
    python run_all.py --model word2vec
    python run_all.py --model transformer
    python run_all.py --model transformer_umls
"""

import json
import argparse
import importlib.util
from pathlib import Path
from datetime import date
import sys

import pandas as pd


# ─── paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'


# ─── team-model loader ────────────────────────────────────────────────────────
# every team model ships as `models/<name>/model.py`. If we imported them as
# plain `from model import X` with sys.path manipulation, Python would cache
# the first-loaded `model` module under sys.modules['model'] and reuse it for
# subsequent loads — a trap for anyone trying to load multiple team models in
# one process. `_import_team_model` loads each one under a unique name.

def _import_team_model(model_dir: Path, class_name: str):
    module_name = f'team_model_{model_dir.name}'
    spec = importlib.util.spec_from_file_location(module_name, model_dir / 'model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


# ─── embedder registry ────────────────────────────────────────────────────────
# add your team's models here as they are pushed to models/

def load_embedder(model_name: str):
    """
    Load and return the correct embedder for a given model name.
    This is the only place you need to add new models.
    """
    if model_name == 'pubmedbert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT / 'models' / 'pubmedbert-local'))
        return embedder
    elif model_name == 'sapbert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT / 'models' / 'sapbert-local'))
        embedder._name = 'sapbert'
        return embedder

    elif model_name == 'biobert':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT / 'models' / 'biobert-local'))
        embedder._name = 'biobert'
        return embedder

    elif model_name == 'minilm':
        from pubmedbert_embedder import PubMedBERTEmbedder
        embedder = PubMedBERTEmbedder()
        embedder.load(str(ROOT / 'models' / 'minilm-local'))
        embedder._name = 'minilm'
        return embedder

    # ── team models ──
    elif model_name == 'word2vec':
        model_dir = ROOT / 'models' / 'word2vec'
        Cls = _import_team_model(model_dir, 'Word2VecEmbedder')
        embedder = Cls()
        embedder.load(str(model_dir))
        embedder._name = 'word2vec'
        return embedder

    elif model_name == 'word2vec_umls':
        model_dir = ROOT / 'models' / 'word2vec_umls'
        Cls = _import_team_model(model_dir, 'Word2VecUMLSEmbedder')
        embedder = Cls()
        embedder.load(str(model_dir))
        embedder._name = 'word2vec_umls'
        return embedder

    elif model_name == 'word2vec_umls_enhanced':
        model_dir = ROOT / 'models' / 'word2vec_umls_enhanced'
        Cls = _import_team_model(model_dir, 'Word2VecUMLSEmbedder')
        embedder = Cls()
        embedder.load(str(model_dir))
        embedder._name = 'word2vec_umls_enhanced'
        return embedder

    elif model_name == 'transformer_fast':
        model_dir = ROOT / 'models' / 'transformer_fast'
        Cls = _import_team_model(model_dir, 'TransformerEmbedder')
        embedder = Cls()
        embedder.load(str(model_dir))
        embedder._name = 'transformer_fast'
        return embedder

    elif model_name == 'transformer_umls_fast':
        model_dir = ROOT / 'models' / 'transformer_umls_fast'
        Cls = _import_team_model(model_dir, 'TransformerUMLSEmbedder')
        embedder = Cls()
        embedder.load(str(model_dir))
        embedder._name = 'transformer_umls_fast'
        return embedder

    elif model_name == 'transformer_umls_enhanced':
        model_dir = ROOT / 'models' / 'transformer_umls_enhanced'
        Cls = _import_team_model(model_dir, 'TransformerUMLSEmbedder')
        embedder = Cls()
        embedder.load(str(model_dir))
        embedder._name = 'transformer_umls_enhanced'
        return embedder

    else:
        raise ValueError(
            f'unknown model: {model_name}\n'
            f'add it to the load_embedder() registry in run_all.py'
        )


# ─── leaderboard builder ──────────────────────────────────────────────────────

def build_leaderboard():
    """
    Read all result JSONs from results/ and compile into leaderboard.csv
    """
    rows = []

    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for result_file in model_dir.glob('*.json'):
            with open(result_file) as f:
                result = json.load(f)

            task = result_file.stem  # e.g. entity_linking_ncbi

            row = {'model': model_name, 'task': task, 'date': result.get('date', '')}

            # entity linking metrics
            if 'acc@1' in result:
                row['acc@1']  = result['acc@1']
                row['acc@5']  = result['acc@5']
                row['mrr']    = result['mrr']

            # sts metrics
            if 'pearson_r' in result:
                row['pearson_r']  = result['pearson_r']
                row['spearman_r'] = result['spearman_r']

            # nli metrics
            if 'accuracy' in result:
                row['accuracy']          = result['accuracy']
                row['macro_f1']          = result['macro_f1']
                row['majority_baseline'] = result['majority_baseline']

            rows.append(row)

    if not rows:
        print('no results found yet — run evaluations first')
        return

    df = pd.DataFrame(rows).sort_values(['task', 'model'])
    leaderboard_path = RESULTS_DIR / 'leaderboard.csv'
    df.to_csv(leaderboard_path, index=False)
    print(f'\nleaderboard saved to {leaderboard_path}')
    print(df.to_string(index=False))
    return df


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='pubmedbert | sapbert | biobert | minilm | '
                             'word2vec | word2vec_umls | word2vec_umls_enhanced | '
                             'transformer_fast | transformer_umls_fast | transformer_umls_enhanced')
    parser.add_argument('--task',  default='all',
                        help='all | entity_linking | sts | nli  (default: all)')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print(f'  model : {args.model}')
    print(f'  task  : {args.task}')
    print(f'{"="*60}\n')

    # load embedder once — reused across all tasks
    embedder = load_embedder(args.model)

    results = {}

    # ── entity linking ──
    if args.task in ('all', 'entity_linking'):
        from eval_entity_linking import evaluate as eval_el

        print('\n--- entity linking: NCBI ---')
        results['ncbi'] = eval_el(
            embedder=embedder, dataset='ncbi',
            batch_size=args.batch_size
        )

        print('\n--- entity linking: BC5CDR-d ---')
        results['bc5cdr_d'] = eval_el(
            embedder=embedder, dataset='bc5cdr_d',
            batch_size=args.batch_size
        )

        print('\n--- entity linking: BC5CDR-c ---')
        results['bc5cdr_c'] = eval_el(
            embedder=embedder, dataset='bc5cdr_c',
            batch_size=args.batch_size
        )

    # ── sts ──
    if args.task in ('all', 'sts'):
        from eval_sts import evaluate as eval_sts

        print('\n--- STS: BIOSSES ---')
        results['biosses'] = eval_sts(
            embedder=embedder, dataset='biosses',
            batch_size=args.batch_size
        )

    # ── nli ──
    if args.task in ('all', 'nli'):
        from eval_nli import evaluate as eval_nli

        print('\n--- NLI: NLI4CT ---')
        results['nli4ct'] = eval_nli(
            embedder=embedder, dataset='nli4ct',
            batch_size=args.batch_size
        )
    

    # ── build leaderboard ──
    print('\n--- building leaderboard ---')
    build_leaderboard()

    print(f'\ndone. all results saved to results/{args.model}/')


if __name__ == '__main__':
    main()
