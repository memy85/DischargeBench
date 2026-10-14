"""Project path resolution shared by the simulation and evaluation code."""
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(_REPO_ROOT / "config/config.yaml") as f:
        return yaml.safe_load(f)


_cfg = load_config()
# `project_path` in config/config.yaml may be null (default: the repository root) or an
# absolute/relative path; relative paths are resolved against the repository root.
PROJECT_PATH = (_REPO_ROOT / _cfg["project_path"]).resolve() if _cfg.get("project_path") else _REPO_ROOT
