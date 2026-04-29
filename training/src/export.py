import json
import shutil
from pathlib import Path

import torch
from gensim.models import KeyedVectors

from .utils import ensure_dir, write_json


WORD2VEC_TEMPLATE = """import json
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


class {class_name}(BaseEmbedder):
    def __init__(self):
        self.wv = None
        self.metadata = None
        self.projection = None
        self._name = '{model_name}'

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
"""


TRANSFORMER_TEMPLATE = """import json
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


class {class_name}(BaseEmbedder):
    def __init__(self):
        self.model = None
        self.projection = None
        self.vocab = None
        self.metadata = None
        self._name = '{model_name}'

    def load(self, model_path):
        meta_path = os.path.join(model_path, 'metadata.json')
        vocab_path = os.path.join(model_path, 'weights', 'vocab.json')
        weights_path = os.path.join(model_path, 'weights', 'transformer.pt')
        with open(meta_path, 'r', encoding='utf-8') as handle:
            self.metadata = json.load(handle)
        with open(vocab_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        tokens = payload['special_tokens'] + payload['tokens']
        self.vocab = {{token: idx for idx, token in enumerate(tokens)}}
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
"""


def export_word2vec_baseline(config: dict, context, keyed_vectors: KeyedVectors) -> None:
    root = Path(context.paths["root"])
    weights = root / "weights"
    ensure_dir(weights)
    keyed_vectors.save_word2vec_format(str(weights / "vectors.bin"), binary=True)
    write_json(root / "metadata.json", {
        "model_name": config["run_name"],
        "model_type": "word2vec",
        "vector_dim": keyed_vectors.vector_size,
        "pooling": "mean",
    })
    (root / "model.py").write_text(
        WORD2VEC_TEMPLATE.format(class_name="Word2VecEmbedder", model_name=config["run_name"]),
        encoding="utf-8",
    )


def export_transformer_baseline(config: dict, context, model, vocab_path: str) -> None:
    root = Path(context.paths["root"])
    weights = root / "weights"
    ensure_dir(weights)
    torch.save(model.state_dict(), weights / "transformer.pt")
    shutil.copyfile(vocab_path, weights / "vocab.json")
    write_json(root / "metadata.json", {
        "model_name": config["run_name"],
        "model_type": "transformer",
        "model_config": config["model"],
        "pooling": config["model"].get("pooling", "cls"),
    })
    (root / "model.py").write_text(
        TRANSFORMER_TEMPLATE.format(class_name="TransformerEmbedder", model_name=config["run_name"]),
        encoding="utf-8",
    )


def export_alignment_model(config: dict, context, artifacts) -> None:
    root = Path(context.paths["root"])
    weights = root / "weights"
    ensure_dir(weights)
    align_cfg = config["alignment"]
    model_type = align_cfg["base_model_type"]

    if model_type == "word2vec":
        kv = KeyedVectors(vector_size=artifacts.adapter.embedding_dim)
        kv.add_vectors(artifacts.adapter.tokens, artifacts.adapter.embedding.weight.detach().cpu().numpy())
        kv.save_word2vec_format(str(weights / "vectors.bin"), binary=True)
        kv.save_word2vec_format(str(weights / "word2vec.bin"), binary=True)
        torch.save(artifacts.head.state_dict(), weights / "projection.pt")
        torch.save(artifacts.head.state_dict(), weights / "projection_head.pt")
        if artifacts.type_classifier is not None:
            torch.save(artifacts.type_classifier.state_dict(), weights / "type_classifier.pt")
        write_json(root / "metadata.json", {
            "model_name": config["run_name"],
            "model_type": "word2vec_umls_enhanced" if artifacts.type_classifier is not None else "word2vec_umls",
            "vector_dim": artifacts.adapter.embedding_dim,
            "projection_dim": align_cfg["projection_dim"],
            "use_projection_at_inference": align_cfg.get("save_projected_inference", True),
            "inference_mode": "projected" if align_cfg.get("save_projected_inference", True) else "base",
            "type_vocab": artifacts.type_payload["type_vocab"] if artifacts.type_payload else None,
        })
        (root / "model.py").write_text(
            WORD2VEC_TEMPLATE.format(class_name="Word2VecUMLSEmbedder", model_name=config["run_name"]),
            encoding="utf-8",
        )
    else:
        torch.save(artifacts.adapter.model.state_dict(), weights / "transformer.pt")
        vocab_src = Path(align_cfg["base_model_dir"]) / "weights" / "vocab.json"
        shutil.copyfile(vocab_src, weights / "vocab.json")
        torch.save(artifacts.head.state_dict(), weights / "projection.pt")
        torch.save(artifacts.head.state_dict(), weights / "projection_head.pt")
        if artifacts.type_classifier is not None:
            torch.save(artifacts.type_classifier.state_dict(), weights / "type_classifier.pt")
        base_meta = Path(align_cfg["base_model_dir"]) / "metadata.json"
        base_payload = json.loads(base_meta.read_text(encoding="utf-8"))
        write_json(root / "metadata.json", {
            "model_name": config["run_name"],
            "model_type": "transformer_umls_enhanced" if artifacts.type_classifier is not None else "transformer_umls",
            "model_config": base_payload["model_config"],
            "projection_dim": align_cfg["projection_dim"],
            "pooling": base_payload.get("pooling", "cls"),
            "use_projection_at_inference": align_cfg.get("save_projected_inference", True),
            "inference_mode": "projected" if align_cfg.get("save_projected_inference", True) else "base",
            "type_vocab": artifacts.type_payload["type_vocab"] if artifacts.type_payload else None,
        })
        (root / "model.py").write_text(
            TRANSFORMER_TEMPLATE.format(class_name="TransformerUMLSEmbedder", model_name=config["run_name"]),
            encoding="utf-8",
        )
