"""Export B1 checkpoints and preserve inference metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from indivoice.models import LogMelCNN, count_parameters

DEFAULT_FEATURE_CONFIG = {
    "sample_rate": 16000,
    "crop_seconds": 4.0,
    "n_fft": 512,
    "hop_length": 160,
    "win_length": 400,
    "n_mels": 80,
    "f_min": 20,
    "f_max": 7600,
    "eps": 1e-6,
}


def _load_model(checkpoint: str | Path, variant: str, device: str = "cpu") -> LogMelCNN:
    model = LogMelCNN(variant=variant).to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    return model


def _ort_logits(path: Path, features: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: features.astype(np.float32)})[0]


def export_b1(
    checkpoint: str | Path,
    output: str | Path,
    variant: str = "small",
    feature_config: dict | None = None,
    temperature: float = 1.0,
    fixed_threshold: float = 0.0,
    eer_threshold: float | None = None,
    opset: int = 17,
    quantize: bool = False,
    dev_npz: str | Path | None = None,
) -> dict:
    """Export B1 and write a sidecar containing all non-weight inference settings."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(checkpoint, variant)
    dummy = torch.zeros(1, 80, 401, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["log_mel"],
        output_names=["logit"],
        dynamic_axes={"log_mel": {0: "batch"}, "logit": {0: "batch"}},
        dynamo=False,
    )
    resolved_features = {**DEFAULT_FEATURE_CONFIG, **(feature_config or {})}
    metadata = {
        "model": "b1_logmel_cnn",
        "variant": variant,
        "parameters": count_parameters(model),
        "input": {"name": "log_mel", "shape": ["batch", 80, 401]},
        "feature_config": resolved_features,
        "calibration": {"temperature": temperature},
        "decision_thresholds": {"fixed": fixed_threshold, "eer": eer_threshold},
        "opset": opset,
        "pytorch_onnx_tolerance": {"rtol": 1e-4, "atol": 1e-5},
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    parity = _ort_logits(output_path, dummy.numpy())
    with torch.no_grad():
        pytorch = model(dummy).numpy()[:, None]
    max_abs_error = float(np.max(np.abs(pytorch - parity)))
    if not np.allclose(pytorch, parity, rtol=1e-4, atol=1e-5):
        raise AssertionError(f"ONNX Runtime output mismatch: max absolute error {max_abs_error}")
    result = {
        "onnx": str(output_path),
        "metadata": str(metadata_path),
        "size_mb": output_path.stat().st_size / 1_000_000,
        "max_abs_error": max_abs_error,
        "quantized": False,
    }
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = output_path.with_name(output_path.stem + ".int8.onnx")
        quantize_dynamic(
            str(output_path),
            str(quantized_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
        result["quantized"] = True
        result["quantized_onnx"] = str(quantized_path)
        if dev_npz is not None:
            arrays = np.load(dev_npz)
            features, labels = arrays["features"].astype(np.float32), arrays["labels"].astype(int)
            unquantized = _ort_logits(output_path, features).reshape(-1)
            quantized = _ort_logits(quantized_path, features).reshape(-1)
            result["dev_accuracy_fp32"] = float(np.mean((unquantized >= fixed_threshold) == labels))
            result["dev_accuracy_int8"] = float(np.mean((quantized >= fixed_threshold) == labels))
            result["dev_accuracy_delta"] = result["dev_accuracy_int8"] - result["dev_accuracy_fp32"]
    return result
