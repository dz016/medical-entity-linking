import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluation'))
from base_embedder import BaseEmbedder


class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, ffn_dim, dropout, max_length):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        return self.encoder(hidden, src_key_padding_mask=(attention_mask == 0))


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


class TransformerUMLSEmbedder(BaseEmbedder):
    def __init__(self):
        self.model = None
        self.projection = None
        self.vocab = None
        self.metadata = None
        self._name = 'transformer_umls_enhanced'

    def load(self, model_path):
        meta_path = os.path.join(model_path, 'metadata.json')
        vocab_path = os.path.join(model_path, 'weights', 'vocab.json')
        weights_path = os.path.join(model_path, 'weights', 'transformer.pt')
        with open(meta_path, 'r', encoding='utf-8') as handle:
            self.metadata = json.load(handle)
        with open(vocab_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        tokens = payload['special_tokens'] + payload['tokens']
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        cfg = self.metadata['model_config']
        self.model = TransformerEncoderModel(
            vocab_size=len(tokens),
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
            ffn_dim=cfg['ffn_dim'],
            dropout=cfg['dropout'],
            max_length=cfg['max_length'],
        )
        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        proj_path = os.path.join(model_path, 'weights', 'projection.pt')
        if os.path.exists(proj_path):
            self.projection = ProjectionHead(cfg['hidden_size'], self.metadata['projection_dim'])
            self.projection.load_state_dict(torch.load(proj_path, map_location='cpu'))
            self.projection.eval()

    def encode(self, texts, batch_size=32):
        mode = self.metadata.get('inference_mode', 'projected' if self.metadata.get('use_projection_at_inference', False) else 'base')
        pad_id = self.vocab['[PAD]']
        cls_id = self.vocab['[CLS]']
        unk_id = self.vocab['[UNK]']
        max_length = self.metadata['model_config']['max_length']
        pooling = self.metadata.get('pooling', 'cls')
        outputs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            ids = []
            masks = []
            for text in chunk:
                seq = [cls_id] + [self.vocab.get(tok, unk_id) for tok in text.lower().split()]
                seq = seq[:max_length]
                ids.append(seq)
            max_len = max(len(seq) for seq in ids)
            for i, seq in enumerate(ids):
                pad = max_len - len(seq)
                masks.append([1] * len(seq) + [0] * pad)
                ids[i] = seq + [pad_id] * pad
            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.tensor(masks, dtype=torch.long)
            hidden = self.model(input_ids, attention_mask)
            if pooling == 'cls':
                pooled = hidden[:, 0]
            else:
                mask = attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            if self.projection is not None and mode == 'projected':
                pooled = self.projection(pooled)
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs).astype(np.float32)

    @property
    def name(self):
        return self._name
