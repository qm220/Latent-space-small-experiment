from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "project.json"


def load_project_config(path: Path | None = None) -> dict:
    cfg_path = path or CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(cfg_path)
    return config


def resolve_under_root(value: str | Path, root: Path | None = None) -> Path:
    root = root or PROJECT_ROOT
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def storage_dir(config: dict | None = None) -> Path:
    config = config or load_project_config()
    return resolve_under_root(config["storage_dir"])


def mongita_dir(config: dict | None = None) -> Path:
    return storage_dir(config) / "mongita_db"
