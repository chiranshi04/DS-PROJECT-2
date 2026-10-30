from pathlib import Path

import numpy as np
import onnx
import soundfile as sf
from fastapi.testclient import TestClient
from onnx import TensorProto, helper

from app.main import create_app


def fixture_model(path: Path) -> None:
    node = helper.make_node("ReduceMean", ["x"], ["y"], axes=[1, 2], keepdims=0)
    graph = helper.make_graph(
        [node],
        "fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 80, 401])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def fixture_wav(tmp_path: Path) -> bytes:
    path = tmp_path / "fixture.wav"
    sf.write(path, np.sin(np.linspace(0, 1000, 16000)).astype("float32"), 16000)
    return path.read_bytes()


def test_upload_returns_schema_and_warnings(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    fixture_model(model)
    client = TestClient(create_app(model_path=model))
    response = client.post(
        "/analyze", files={"file": ("fixture.wav", fixture_wav(tmp_path), "audio/wav")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert {
        "score",
        "band",
        "per_window_scores",
        "duration_seconds",
        "warnings",
        "limitations",
        "waveform",
    } <= payload.keys()
    assert any("Language not verified" in warning for warning in payload["warnings"])


def test_model_missing_is_explicit(tmp_path: Path) -> None:
    client = TestClient(create_app(model_path=tmp_path / "missing.onnx"))
    assert client.get("/health").json()["model_available"] is False
    response = client.post("/analyze", files={"file": ("fixture.wav", b"bad", "audio/wav")})
    assert response.status_code == 503
    assert "Model unavailable" in response.json()["detail"]


def test_invalid_extension_and_oversize_are_clear(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    fixture_model(model)
    client = TestClient(create_app(model_path=model))
    invalid = client.post("/analyze", files={"file": ("fixture.txt", b"hello", "text/plain")})
    assert invalid.status_code == 415
    client.app.state.service.max_bytes = 4
    oversized = client.post("/analyze", files={"file": ("fixture.wav", b"12345", "audio/wav")})
    assert oversized.status_code == 413


def test_limitations_text_is_always_present(tmp_path: Path) -> None:
    client = TestClient(create_app(model_path=tmp_path / "missing.onnx"))
    html = client.get("/").text
    assert "research demo" in html
    assert "synthetic speech does not mean fraud" in html
