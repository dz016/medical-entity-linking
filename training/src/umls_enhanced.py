import json
from collections import Counter, defaultdict
from pathlib import Path

from gensim.models import KeyedVectors

from .utils import ensure_dir, write_json


ALLOWED_RELATIONS = {
    "has_manifestation",
    "may_treat",
    "treated_by",
    "finding_site_of",
}


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _term_in_vocab(text: str, vocab: set[str]) -> bool:
    return any(token in vocab for token in _normalize_text(text).split())


def load_encoder_vocab(vocab_source_path: str) -> set[str]:
    path = Path(vocab_source_path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        special = payload.get("special_tokens", [])
        tokens = payload.get("tokens", [])
        return {token for token in [*special, *tokens] if token}
    kv = KeyedVectors.load_word2vec_format(str(path), binary=True)
    return set(kv.key_to_index.keys())


def load_mrconso_maps(mrconso_path: str, vocab: set[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    cui_to_terms: dict[str, list[str]] = defaultdict(list)
    term_to_cuis: dict[str, list[str]] = defaultdict(list)
    seen = set()
    with Path(mrconso_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue
            cui, lat, _, _, _, _, _, _, _, _, _, _, _, _, text = parts[:15]
            if lat != "ENG":
                continue
            text = _normalize_text(text)
            if not text or not _term_in_vocab(text, vocab):
                continue
            key = (cui, text)
            if key in seen:
                continue
            seen.add(key)
            cui_to_terms[cui].append(text)
            term_to_cuis[text].append(cui)
    return dict(cui_to_terms), dict(term_to_cuis)


def build_cui_to_type(
    mrsty_path: str,
    mrconso_path: str,
    vocab_source_path: str,
    output_path: str,
    max_types: int = 30,
) -> dict:
    vocab = load_encoder_vocab(vocab_source_path)
    cui_to_terms, term_to_cuis = load_mrconso_maps(mrconso_path, vocab)
    cui_types: dict[str, list[str]] = defaultdict(list)
    type_counter: Counter[str] = Counter()

    with Path(mrsty_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            cui = parts[0]
            sty = parts[3].strip()
            if cui not in cui_to_terms or not sty:
                continue
            cui_types[cui].append(sty)
            type_counter[sty] += 1

    type_vocab = [sty for sty, _ in type_counter.most_common(max_types)]
    type_to_index = {sty: idx for idx, sty in enumerate(type_vocab)}
    class_weights = {}
    total = sum(type_counter[sty] for sty in type_vocab) or 1
    for sty in type_vocab:
        freq = type_counter[sty]
        class_weights[sty] = float(total / max(freq, 1))

    payload = {
        "type_vocab": type_vocab,
        "type_to_index": type_to_index,
        "class_weights": class_weights,
        "cui_to_types": {
            cui: sorted({sty for sty in types if sty in type_to_index})
            for cui, types in cui_types.items()
            if any(sty in type_to_index for sty in types)
        },
        "term_to_cuis": term_to_cuis,
        "cui_to_terms": cui_to_terms,
    }
    write_json(output_path, payload)
    return payload


def build_relation_pairs(
    mrrel_path: str,
    mrconso_path: str,
    vocab_source_path: str,
    output_path: str,
    relation_types: list[str] | None = None,
) -> list[dict]:
    relation_types = set(relation_types or ALLOWED_RELATIONS)
    vocab = load_encoder_vocab(vocab_source_path)
    cui_to_terms, _ = load_mrconso_maps(mrconso_path, vocab)
    pairs = []
    seen = set()
    with Path(mrrel_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8:
                continue
            cui1 = parts[0]
            rel = parts[3].strip().lower()
            cui2 = parts[4]
            rela = parts[7].strip().lower()
            relation = rela or rel
            if relation not in relation_types:
                continue
            if cui1 not in cui_to_terms or cui2 not in cui_to_terms:
                continue
            anchor_text = cui_to_terms[cui1][0]
            positive_text = cui_to_terms[cui2][0]
            if not anchor_text or not positive_text or anchor_text == positive_text:
                continue
            key = (cui1, relation, cui2, anchor_text, positive_text)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "anchor_cui": cui1,
                    "anchor_text": anchor_text,
                    "relation_type": relation,
                    "positive_cui": cui2,
                    "positive_text": positive_text,
                }
            )
    ensure_dir(Path(output_path).parent)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(pairs, handle, indent=2)
    return pairs
