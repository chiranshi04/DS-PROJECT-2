"""Run configured training with ``python -m indivoice.train --config ...``."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from indivoice.config import config_as_dict
from indivoice.data.dataset import ManifestAudioDataset, make_dataloader
from indivoice.features.mel import LogMelExtractor
from indivoice.models import build_model
from indivoice.train.engine import fit_model
from indivoice.utils.seed import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    seed_everything(int(cfg.get("seed", 2025)))
    if cfg.get("model", "b1") == "b0":
        raise SystemExit(
            "B0 is a classical baseline; use its dedicated feature-fitting script "
            "in a later evaluation phase."
        )
    train_manifest, dev_manifest = cfg.get("train_manifest"), cfg.get("dev_manifest")
    if (
        not train_manifest
        or not dev_manifest
        or not Path(str(train_manifest)).exists()
        or not Path(str(dev_manifest)).exists()
    ):
        raise SystemExit(
            "Training requires verified train_manifest and dev_manifest paths; "
            "no real-data run was started."
        )
    feature_config = config_as_dict("features")
    mel_keys = {
        "sample_rate",
        "n_fft",
        "hop_length",
        "win_length",
        "n_mels",
        "f_min",
        "f_max",
        "eps",
    }
    features = LogMelExtractor(
        **{key: feature_config[key] for key in mel_keys if key in feature_config}
    )
    train_set = ManifestAudioDataset(
        str(train_manifest), features, training=True, seed=int(cfg.get("seed", 2025))
    )
    dev_set = ManifestAudioDataset(
        str(dev_manifest), features, training=False, seed=int(cfg.get("seed", 2025))
    )
    train_loader = make_dataloader(
        train_set, int(cfg.get("batch_size", 16)), int(cfg.get("num_workers", 0))
    )
    dev_loader = make_dataloader(
        dev_set, int(cfg.get("batch_size", 16)), int(cfg.get("num_workers", 0)), shuffle=False
    )
    model = build_model(str(cfg.get("model", "b1")), str(cfg.get("variant", "small")))
    run_dir = Path(str(cfg.get("run_dir", "runs/default")))
    fit_model(model, train_loader, dev_loader, dict(cfg), run_dir)
    print(f"Training complete; artifacts written to {run_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
