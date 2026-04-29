import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from gensim.models import KeyedVectors
from tqdm import tqdm

from .utils import ensure_dir, write_json


class SimpleTokenizer:
    _abstract_re = re.compile(r"<AbstractText[^>]*>(.*?)</AbstractText>", re.DOTALL)
    _sent_split_re = re.compile(r"(?<=[.!?])\s+")
    _strip_re = re.compile(r"[^a-z0-9]+")

    def __init__(self, min_token_length: int = 2):
        self.min_token_length = min_token_length

    def tokenize_text(self, text: str) -> list[str]:
        text = text.lower()
        tokens = []
        for raw in text.split():
            token = self._strip_re.sub("", raw)
            if len(token) >= self.min_token_length:
                tokens.append(token)
        return tokens

    def iter_pubmed_sentences(self, pubmed_dir: str, max_sentences: int | None = None):
        files = sorted(Path(pubmed_dir).glob("**/*.xml.gz"))
        seen = 0
        for xml_path in tqdm(files, desc="PubMed XML", unit="file"):
            with gzip.open(xml_path, "rt", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
            for match in self._abstract_re.finditer(content):
                abstract = re.sub(r"<[^>]+>", " ", match.group(1))
                for sent in self._sent_split_re.split(abstract):
                    tokens = self.tokenize_text(sent)
                    if tokens:
                        yield tokens
                        seen += 1
                        if max_sentences is not None and seen >= max_sentences:
                            return


def materialize_corpus(pubmed_dir: str, output_path: str, tokenizer: SimpleTokenizer, max_sentences: int | None = None) -> None:
    ensure_dir(Path(output_path).parent)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for tokens in tokenizer.iter_pubmed_sentences(pubmed_dir, max_sentences=max_sentences):
            handle.write(json.dumps(tokens) + "\n")


class TokenizedCorpus:
    def __init__(self, path: str):
        self.path = Path(path)

    def __iter__(self):
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def load_tokenized_corpus(path: str) -> TokenizedCorpus:
    return TokenizedCorpus(path)


def build_vocabulary(corpus_path: str, vocab_path: str, min_freq: int, max_vocab_size: int | None = None) -> dict:
    counter = Counter()
    for tokens in load_tokenized_corpus(corpus_path):
        counter.update(tokens)

    items = [(token, freq) for token, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda item: (-item[1], item[0]))
    if max_vocab_size is not None:
        items = items[:max_vocab_size]

    vocab = {
        "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[MASK]"],
        "tokens": [token for token, _ in items],
        "frequencies": {token: freq for token, freq in items},
    }
    write_json(vocab_path, vocab)
    return vocab


def extract_umls_pairs(
    mrconso_path: str,
    output_pairs_path: str,
    language: str = "ENG",
    max_pairs_per_cui: int = 10,
    keyed_vectors_path: str | None = None,
) -> None:
    vocab = None
    if keyed_vectors_path is not None:
        kv = KeyedVectors.load_word2vec_format(keyed_vectors_path, binary=True)
        vocab = set(kv.key_to_index)

    cui_to_strings: dict[str, set[str]] = defaultdict(set)
    with Path(mrconso_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in tqdm(handle, desc="MRCONSO", unit="row"):
            cols = line.rstrip("\n").split("|")
            if len(cols) < 17:
                continue
            cui, lat, is_pref, string, suppress = cols[0], cols[1], cols[6], cols[14], cols[16]
            if lat != language or suppress != "N":
                continue
            normalized = " ".join(string.lower().split())
            if not normalized:
                continue
            cui_to_strings[cui].add(normalized)

    ensure_dir(Path(output_pairs_path).parent)
    with Path(output_pairs_path).open("w", encoding="utf-8") as handle:
        for strings in tqdm(cui_to_strings.values(), desc="Pair extraction", unit="cui"):
            values = sorted(strings)
            if len(values) < 2:
                continue
            emitted = 0
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    a, b = values[i], values[j]
                    if vocab is not None:
                        if not any(tok in vocab for tok in a.split()) or not any(tok in vocab for tok in b.split()):
                            continue
                    handle.write(f"{a}\t{b}\n")
                    emitted += 1
                    if emitted >= max_pairs_per_cui:
                        break
                if emitted >= max_pairs_per_cui:
                    break
