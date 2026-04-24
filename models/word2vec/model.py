import json
import os
import sys
import numpy as np
from gensim.models import KeyedVectors

sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))
from base_embedder import BaseEmbedder


class Word2VecEmbedder(BaseEmbedder):
    def __init__(self):
        self.wv = None
        self.metadata = None
        self._name = 'word2vec'

    def load(self, model_path):
        weights_path = os.path.join(model_path, 'weights', 'vectors.bin')
        meta_path = os.path.join(model_path, 'metadata.json')
        self.wv = KeyedVectors.load_word2vec_format(weights_path, binary=True)
        with open(meta_path, 'r', encoding='utf-8') as handle:
            self.metadata = json.load(handle)

    def encode(self, texts, batch_size=32):
        outputs = []
        for text in texts:
            toks = [tok for tok in text.lower().split() if tok in self.wv]
            if not toks:
                outputs.append(np.zeros(self.wv.vector_size, dtype=np.float32))
            else:
                outputs.append(np.mean([self.wv[tok] for tok in toks], axis=0).astype(np.float32))
        return np.vstack(outputs).astype(np.float32)

    @property
    def name(self):
        return self._name
