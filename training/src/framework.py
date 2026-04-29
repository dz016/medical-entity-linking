import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from .utils import artifact_paths, ensure_dir, write_json


log = logging.getLogger(__name__)


@dataclass
class RunContext:
    run_name: str
    output_root: str
    config: dict
    resume: bool = False

    def __post_init__(self):
        self.paths = artifact_paths(self.output_root)
        ensure_dir(self.paths["root"])
        write_json(Path(self.paths["root"]) / "config_snapshot.json", self.config)

    def checkpoint_path(self, name: str) -> str:
        return str(Path(self.paths["checkpoints"]) / name)

    def save_checkpoint(self, payload: dict, name: str = "latest.pt") -> None:
        path = self.checkpoint_path(name)
        torch.save(payload, path)
        log.info("Saved checkpoint -> %s", path)

    def load_checkpoint(self, name: str = "latest.pt") -> dict | None:
        path = Path(self.checkpoint_path(name))
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu")

