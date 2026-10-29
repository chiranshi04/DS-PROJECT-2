"""Phase 1 smoke tests."""

import random

import numpy as np

from indivoice import __version__
from indivoice.config import load_config
from indivoice.utils.seed import seed_everything


def test_package_import_and_config() -> None:
    assert __version__ == "0.1.0"
    assert load_config("data").languages == ["hi", "ta", "bn"]


def test_seed_is_deterministic() -> None:
    seed_everything(123)
    first = (random.random(), float(np.random.rand()))
    seed_everything(123)
    second = (random.random(), float(np.random.rand()))
    assert first == second
