import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))

import numpy as np
from gensim.models import KeyedVectors
from base_embedder import BaseEmbedder


class Word2VecEmbedder(BaseEmbedder):
    """
    Word2Vec embedder trained with XCENT contrastive loss on PubMed + UMLS.
    Same architecture as plain Word2Vec — UMLS knowledge is baked into
    the vectors during training, no projection needed at inference.
    Encodes text by mean-pooling over in-vocabulary word vectors.
    OOV tokens are silently skipped; all-OOV inputs return a zero vector.

    Vectors are L2-normalized after loading to prevent overflow during
    cosine similarity computation.

    Note: binary file has a corrupted entry around word 80k-100k.
    limit=80000 loads all clean vectors safely.
    """

    def __init__(self):
        self.wv    = None
        self._name = 'word2vec_umls'

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        """
        Load word vectors from the weights folder.
        Expects: <model_path>/weights/word2vec_umls.bin
        Note: projection.pt is NOT loaded — it was used only during
        contrastive training, not needed for inference.
        """
        weights_file = os.path.join(model_path, 'weights', 'word2vec_umls.bin')
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f'weights not found at {weights_file}\n'
                f'Download from shared Drive and place in models/word2vec_umls/weights/'
            )

        print(f'[word2vec_umls] loading from {weights_file} ...')
        self.wv = KeyedVectors.load_word2vec_format(
            weights_file,
            binary=True,
            unicode_errors='ignore',
            limit=80000     # file has corrupted entry after ~80k words
        )
        print(f'[word2vec_umls] loaded — vocab: {len(self.wv):,}  dim: {self.wv.vector_size}')

        # L2-normalize all vectors to unit length
        # prevents overflow during cosine similarity and mean pooling
        print(f'[word2vec_umls] normalizing vectors...')
        self.wv.fill_norms(force=True)
        norms = self.wv.norms[:, np.newaxis]
        norms = np.where(norms == 0, 1.0, norms)
        self.wv.vectors = (self.wv.vectors / norms).astype(np.float32)

        print(f'[word2vec_umls] ready — vocab: {len(self.wv):,}  dim: {self.wv.vector_size}')

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _embed_one(self, text: str) -> np.ndarray:
        """
        Mean-pool word vectors for a single text string.
        Lowercases and whitespace-splits before lookup.
        Returns zero vector for fully OOV input or non-finite result.
        """
        tokens  = text.lower().split()
        vectors = [self.wv[t] for t in tokens if t in self.wv]

        if not vectors:
            return np.zeros(self.wv.vector_size, dtype=np.float32)

        result = np.mean(vectors, axis=0).astype(np.float32)

        if not np.isfinite(result).all():
            return np.zeros(self.wv.vector_size, dtype=np.float32)

        return result

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert N text strings to (N, vector_size) float32 numpy array.
        batch_size is accepted for API compatibility but ignored.
        """
        if self.wv is None:
            raise RuntimeError('call load() before encode()')

        embeddings = [self._embed_one(t) for t in texts]
        return np.vstack(embeddings).astype(np.float32)

    # ------------------------------------------------------------------
    # name property
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name