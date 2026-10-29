"""Configuration loading for reproducible experiments."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> DictConfig:
    """Load one named YAML configuration from ``config_dir``.

    Args:
        name: File stem or filename, for example ``data`` or ``data.yaml``.
        config_dir: Directory containing versioned YAML files.

    Raises:
        FileNotFoundError: If the requested configuration does not exist.
        ValueError: If ``name`` attempts path traversal.
    """
    filename = f"{name}.yaml" if not name.endswith(".yaml") else name
    path = Path(config_dir) / filename
    if Path(filename).name != filename:
        raise ValueError("Configuration name must be a simple filename")
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    return OmegaConf.load(path)


def config_as_dict(name: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    """Load a YAML configuration and convert it to a plain dictionary."""
    return dict(OmegaConf.to_container(load_config(name, config_dir), resolve=True))
