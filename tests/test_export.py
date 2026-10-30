from pathlib import Path

import numpy as np
import pytest
import torch

from indivoice.models import LogMelCNN

ort = pytest.importorskip("onnxruntime")

from indivoice.export.exporter import export_b1  # noqa: E402


def fixture_checkpoint(path: Path) -> None:
    torch.manual_seed(12)
    torch.save({"model": LogMelCNN("tiny").state_dict()}, path)


def test_b1_onnx_matches_pytorch_and_writes_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    output = tmp_path / "b1.onnx"
    fixture_checkpoint(checkpoint)
    result = export_b1(
        checkpoint,
        output,
        variant="tiny",
        feature_config={"sample_rate": 16000},
        temperature=1.7,
        fixed_threshold=0.2,
    )
    assert result["max_abs_error"] <= 1e-5
    assert Path(result["metadata"]).exists()
    metadata = Path(result["metadata"]).read_text(encoding="utf-8")
    assert '"temperature": 1.7' in metadata
    assert '"sample_rate": 16000' in metadata


def test_dynamic_int8_reports_dev_accuracy_delta(tmp_path: Path) -> None:
    checkpoint, output = tmp_path / "best.pt", tmp_path / "b1.onnx"
    fixture_checkpoint(checkpoint)
    features = np.zeros((4, 80, 401), dtype=np.float32)
    labels = np.array([0, 0, 0, 0])
    dev = tmp_path / "dev.npz"
    np.savez(dev, features=features, labels=labels)
    try:
        result = export_b1(checkpoint, output, variant="tiny", quantize=True, dev_npz=dev)
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"optional ONNX quantization unavailable: {error}")
    assert result["quantized"] is True
    assert "dev_accuracy_delta" in result
    assert Path(result["quantized_onnx"]).exists()
