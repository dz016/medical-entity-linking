import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


log = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )


def artifact_paths(root: str) -> dict[str, str]:
    base = Path(root)
    weights = base / "weights"
    checkpoints = base / "checkpoints"
    ensure_dir(weights)
    ensure_dir(checkpoints)
    return {
        "root": str(base),
        "weights": str(weights),
        "checkpoints": str(checkpoints),
        "metadata": str(base / "metadata.json"),
        "model_py": str(base / "model.py"),
    }

