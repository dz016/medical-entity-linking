import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))
from base_embedder import BaseEmbedder


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Word2VecUMLSEmbedder(BaseEmbedder):
    def __init__(self):
        self.wv = None
        self.metadata = None
        self.projection = None
        self._name = 'word2vec_umls_enhanced'

    def load(self, model_path):
        weights_path = os.path.join(model_path, 'weights', 'vectors.bin')
        meta_path = os.path.join(model_path, 'metadata.json')
        self.wv = KeyedVectors.load_word2vec_format(weights_path, binary=True)
        with open(meta_path, 'r', encoding='utf-8') as handle:
            self.metadata = json.load(handle)
        proj_path = os.path.join(model_path, 'weights', 'projection.pt')
        if os.path.exists(proj_path):
            self.projection = ProjectionHead(self.wv.vector_size, self.metadata['projection_dim'])
            self.projection.load_state_dict(torch.load(proj_path, map_location='cpu'))
            self.projection.eval()

    def encode(self, texts, batch_size=32):
        outputs = []
        mode = self.metadata.get('inference_mode', 'projected' if self.metadata.get('use_projection_at_inference', False) else 'base')
        for text in texts:
            toks = [tok for tok in text.lower().split() if tok in self.wv]
            if not toks:
                if self.projection is not None and mode == 'projected':
                    vector = np.zeros(self.metadata['projection_dim'], dtype=np.float32)
                else:
                    vector = np.zeros(self.wv.vector_size, dtype=np.float32)
            else:
                vector = np.mean([self.wv[tok] for tok in toks], axis=0).astype(np.float32)
                if self.projection is not None and mode == 'projected':
                    tensor = torch.from_numpy(vector).unsqueeze(0)
                    vector = self.projection(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
            outputs.append(vector)
        return np.vstack(outputs).astype(np.float32)

    @property
    def name(self):
        return self._name
