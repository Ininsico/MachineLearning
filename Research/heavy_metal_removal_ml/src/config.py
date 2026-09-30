"""Configuration loading, path resolution and global reproducibility controls."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


class DotDict(dict):
    """Dictionary with attribute access for nested configuration nodes.

    Nested mappings are converted once at load time so that attribute access
    returns the *stored* node. This matters because in-code overrides (for
    example CLI flags such as ``cfg.model.grid.enabled = False``) must mutate the
    live configuration rather than a temporary copy.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _to_dotdict(value)


def _to_dotdict(obj: Any) -> Any:
    if isinstance(obj, DotDict):
        return obj
    if isinstance(obj, dict):
        return DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    return obj


def load_config(path: str | os.PathLike[str] | None = None) -> DotDict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cfg = _to_dotdict(raw)
    cfg["_config_path"] = str(cfg_path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(cfg: DotDict, *parts: str) -> Path:
    """Resolve a path relative to the project root and create parent dirs."""
    path = Path(cfg["_project_root"]).joinpath(*parts)
    return path


def ensure_dirs(cfg: DotDict) -> Dict[str, Path]:
    root = Path(cfg["_project_root"])
    paths = {
        "project": root,
        "data": root / cfg.project.data_dir,
        "raw": root / cfg.project.data_dir / "raw",
        "interim": root / cfg.project.data_dir / "interim",
        "processed": root / cfg.project.data_dir / "processed",
        "literature": root / cfg.project.data_dir / "literature",
        "results": root / cfg.project.results_dir,
        "figures": root / cfg.project.results_dir / "figures",
        "tables": root / cfg.project.results_dir / "tables",
        "models": root / cfg.project.results_dir / "models",
        "logs": root / cfg.project.results_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_global_seed(seed: int) -> None:
    """Seed every RNG that influences the pipeline.

    Note: PYTHONHASHSEED must be set before interpreter start to be effective;
    we set it here as well so child processes inherit a deterministic value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def config_fingerprint(cfg: DotDict) -> str:
    """Stable hash of the configuration for provenance in saved artifacts."""
    payload = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
