"""Reproducibility helpers."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch and request deterministic operations."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        # The utility remains usable for lightweight tooling without Torch installed.
        pass
