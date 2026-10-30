"""Small reproducible training loop for binary speech classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from indivoice.train.metrics import equal_error_rate


def _batch_to_device(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["features"].to(device), batch["label"].to(device).float()
    return batch[0].to(device), batch[1].to(device).float()


def _evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses, scores, labels = [], [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            x, y = _batch_to_device(batch, device)
            logits = model(x)
            losses.append(float(criterion(logits, y)))
            scores.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return float(np.mean(losses)), equal_error_rate(np.asarray(labels), np.asarray(scores))


def fit_model(
    model: nn.Module, train_loader, dev_loader, config: dict[str, Any], run_dir: str | Path
) -> list[dict[str, float]]:
    """Train a binary model with weighted BCE, cosine decay, AMP, clipping, and early stopping."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    pos_weight = torch.tensor([float(config.get("pos_weight", 1.0))], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.get("epochs", 10))
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_amp = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_eer, bad_epochs, patience = float("inf"), 0, int(config.get("patience", 5))
    history: list[dict[str, float]] = []
    log_path = run_path / "metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_loss", "dev_loss", "dev_eer", "learning_rate"]
        )
        writer.writeheader()
        for epoch in range(1, int(config.get("epochs", 10)) + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                x, y = _batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.get("gradient_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.detach()))
            scheduler.step()
            dev_loss, dev_eer = _evaluate(model, dev_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "dev_loss": dev_loss,
                "dev_eer": dev_eer,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            writer.writerow(row)
            handle.flush()
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": row}, run_path / "last.pt"
            )
            if dev_eer < best_eer:
                best_eer, bad_epochs = dev_eer, 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "metrics": row},
                    run_path / "best.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
    return history
