"""Config and project path shared by the simulation, evaluation and analysis code."""
from pathlib import Path

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """config/config.yaml with ${...} interpolations resolved."""
    return OmegaConf.to_container(OmegaConf.load(_REPO_ROOT / "config/config.yaml"), resolve=True)


CONFIG = load_config()
# `project_path` in config/config.yaml may be null (default: the repository root) or an
# absolute/relative path; relative paths are resolved against the repository root.
PROJECT_PATH = (_REPO_ROOT / CONFIG["project_path"]).resolve() if CONFIG.get("project_path") else _REPO_ROOT
PROMPTS_PATH = PROJECT_PATH / "prompts"
