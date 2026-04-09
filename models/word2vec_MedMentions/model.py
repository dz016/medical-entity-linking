import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../evaluation"))

import numpy as np
from gensim.models import KeyedVectors
from base_embedder import BaseEmbedder


class Word2VecMedMentionsEmbedder(BaseEmbedder):
    def __init__(self):
        self.wv = None
        self._name = "word2vec_MedMentions"

    def load(self, model_path: str) -> None:
        weights_file = os.path.join(model_path, "weights", "weights.bin")
        print(f"Loading from {weights_file}...")
        self.wv = KeyedVectors.load_word2vec_format(weights_file, binary=True)
        print(f"Loaded — vocab: {len(self.wv)}, dim: {self.wv.vector_size}")

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = text.lower().strip().split()
        vectors = [self.wv[t] for t in tokens if t in self.wv]

        if not vectors:
            return np.zeros(self.wv.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        if self.wv is None:
            raise RuntimeError("Call load() before encode()")
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)

    @property
    def name(self) -> str:
        return self._name