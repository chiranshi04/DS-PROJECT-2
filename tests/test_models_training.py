from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from indivoice.models import LFCCGMM, LogMelCNN, RawWaveTiny, count_parameters
from indivoice.train.engine import fit_model


def test_b1_variants_are_lightweight() -> None:
    tiny, small = LogMelCNN("tiny"), LogMelCNN("small")
    assert count_parameters(tiny) < 1_000_000
    assert count_parameters(small) < 1_000_000
    assert tiny(torch.randn(2, 80, 32)).shape == (2,)
    assert small(torch.randn(2, 80, 32)).shape == (2,)


def test_b2_raw_waveform_shape() -> None:
    assert RawWaveTiny()(torch.randn(2, 16000)).shape == (2,)


def test_b0_lfcc_gmm_likelihood_ratio() -> None:
    rng = np.random.default_rng(3)
    genuine, synthetic = rng.normal(-1, 0.1, (8, 4)), rng.normal(1, 0.1, (8, 4))
    model = LFCCGMM(n_components=2).fit(
        np.vstack([genuine, synthetic]), np.array([0] * 8 + [1] * 8)
    )
    scores = model.score_samples(np.vstack([genuine, synthetic]))
    assert scores[:8].mean() < scores[8:].mean()


def test_tiny_end_to_end_training_reduces_loss(tmp_path: Path) -> None:
    torch.manual_seed(4)
    genuine = torch.zeros(8, 80, 16)
    synthetic = torch.ones(8, 80, 16)
    x = torch.cat([genuine, synthetic])
    y = torch.tensor([0] * 8 + [1] * 8, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = LogMelCNN("tiny")
    history = fit_model(
        model,
        loader,
        loader,
        {
            "device": "cpu",
            "epochs": 3,
            "learning_rate": 0.001,
            "patience": 3,
            "mixed_precision": False,
            "pos_weight": 1.0,
        },
        tmp_path / "run",
    )
    assert len(history) >= 2
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "metrics.csv").exists()
    assert (tmp_path / "run" / "config.json").exists()
