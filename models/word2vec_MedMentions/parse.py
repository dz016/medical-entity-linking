"""
parse.py
--------
Reads the MedMentions PubTator corpus and yields tokenised sentences
ready for Word2Vec training.

PubTator format (one document block per abstract, blank-line separated):

    25763772|t|DCTN4 as a modifier of chronic Pseudomonas...
    25763772|a|Pseudomonas aeruginosa (Pa) infects the lungs...
    25763772\t0\t5\tDCTN4\tT116,T123\tC4308010
    ...
    <blank line>

For Option A we only need the |t| and |a| lines. The annotation rows
(tab-separated) are skipped entirely — they become useful in Option B.

Usage (as a library):
    from parse import iter_sentences
    for tokens in iter_sentences("data/corpus_pubtator.txt"):
        ...  # list[str]

Usage (standalone — preview):
    python parse.py data/corpus_pubtator.txt
"""

import re
import sys
from pathlib import Path
from typing import Iterator

# Matches title and abstract lines: "<PMID>|t|..." or "<PMID>|a|..."
_TEXT_RE = re.compile(r"^\d+\|[ta]\|(.+)$")


def _tokenise(text: str) -> list[str]:
    """
    Lowercase and split on non-alphanumeric boundaries.
    Keeps hyphenated compounds (e.g. 'dose-response') intact,
    and drops pure-punctuation tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    return tokens


def iter_documents(corpus_path: Path) -> Iterator[dict]:
    """
    Yields one dict per PubMed abstract:
        {"pmid": str, "title": str, "abstract": str}
    """
    pmid, title, abstract = None, "", ""

    with open(corpus_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if not line:
                # blank line = document boundary
                if pmid is not None:
                    yield {"pmid": pmid, "title": title, "abstract": abstract}
                pmid, title, abstract = None, "", ""
                continue

            m = _TEXT_RE.match(line)
            if m:
                doc_id, kind = line.split("|", 2)[:2]
                text = m.group(1)
                if pmid is None:
                    pmid = doc_id
                if kind == "t":
                    title = text
                elif kind == "a":
                    abstract = text
            # else: annotation line (tab-separated) — skip for Option A

    # flush last document if file doesn't end with a blank line
    if pmid is not None:
        yield {"pmid": pmid, "title": title, "abstract": abstract}


def iter_sentences(corpus_path: Path) -> Iterator[list[str]]:
    """
    Yields tokenised sentences (list[str]) for every title and abstract.
    Each title is treated as one sentence; each abstract as one sentence.
    This is intentional for Word2Vec — context windows stay within a
    single continuous piece of text.
    """
    for doc in iter_documents(corpus_path):
        if doc["title"]:
            tokens = _tokenise(doc["title"])
            if tokens:
                yield tokens
        if doc["abstract"]:
            tokens = _tokenise(doc["abstract"])
            if tokens:
                yield tokens


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python parse.py <corpus_pubtator.txt>")

    path = Path(sys.argv[1])
    doc_count = sent_count = token_count = 0

    for i, doc in enumerate(iter_documents(path)):
        doc_count += 1

    for tokens in iter_sentences(path):
        sent_count += 1
        token_count += len(tokens)

    print(f"Documents : {doc_count:,}")
    print(f"Sentences : {sent_count:,}  (title + abstract per doc)")
    print(f"Tokens    : {token_count:,}")
    print(f"Avg tok/s : {token_count / sent_count:.1f}")