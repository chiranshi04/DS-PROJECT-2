"""InDivoice CPU research demo web application."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from indivoice.data.audio import AudioDecodeError, load_audio, make_windows
from indivoice.features.mel import LogMelExtractor

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
LIMITATIONS = (
    "This is a research demo; a result does not prove that audio is synthetic or genuine; "
    "synthetic speech does not mean fraud; genuine speech does not mean trustworthy; "
    "the model may fail on unseen voice generators, languages, and heavy noise or compression; "
    "do not use it as the sole basis for any decision."
)


class InferenceService:
    """Load one ONNX model and keep all uploaded audio request-local."""

    def __init__(
        self, model_path: str | Path | None = None, config_path: str | Path = "configs/app.yaml"
    ) -> None:
        config = (
            yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            if Path(config_path).exists()
            else {}
        )
        self.model_path = Path(
            model_path
            or os.environ.get("INDIVOICE_MODEL_PATH", config.get("model_path", "releases/b1.onnx"))
        )
        self.max_bytes = int(float(config.get("max_upload_size_mb", 20)) * 1_000_000)
        self.max_seconds = float(config.get("max_seconds", 60))
        self.temperature = 1.0
        self.fixed_threshold = float(config.get("fixed_threshold", 0.0))
        self.band_lower = float(config.get("band_lower", -0.5))
        self.band_upper = float(config.get("band_upper", 0.5))
        self.extractor = LogMelExtractor()
        self.session = None
        self.model_error = None
        metadata_path = self.model_path.with_suffix(".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.temperature = float(metadata.get("calibration", {}).get("temperature", 1.0))
            self.fixed_threshold = float(
                metadata.get("decision_thresholds", {}).get("fixed", self.fixed_threshold)
            )
        if self.model_path.exists():
            try:
                import onnxruntime as ort

                self.session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
            except Exception as error:  # pragma: no cover - depends on local runtime/provider
                self.model_error = str(error)
        else:
            self.model_error = f"Model not found at {self.model_path}"

    @property
    def available(self) -> bool:
        return self.session is not None and self.model_error is None

    def analyze(self, audio_bytes: bytes, suffix: str) -> dict:
        if not self.available:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {self.model_error}")
        if len(audio_bytes) > self.max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {self.max_bytes / 1_000_000:.0f} MB upload limit",
            )
        if suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Unsupported audio format; accepted formats are WAV, MP3, FLAC, OGG, "
                    "M4A, and OPUS"
                ),
            )
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temporary:
            temporary.write(audio_bytes)
            temporary.flush()
            try:
                clip = load_audio(temporary.name, target_sample_rate=16000)
            except AudioDecodeError as error:
                raise HTTPException(
                    status_code=400, detail=f"Could not decode audio: {error}"
                ) from error
        if clip.original_duration_s > self.max_seconds:
            raise HTTPException(
                status_code=413,
                detail=f"Audio is longer than the {self.max_seconds:.0f} second limit",
            )
        windows = make_windows(clip.waveform, clip.sample_rate, training=False)
        input_name = self.session.get_inputs()[0].name
        logits = []
        for window in windows:
            features = self.extractor(window.waveform).unsqueeze(0).numpy().astype(np.float32)
            raw_logit = self.session.run(None, {input_name: features})[0]
            logits.append(float(np.asarray(raw_logit).reshape(-1)[0]) / self.temperature)
        probabilities = [float(1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))) for logit in logits]
        score = float(np.mean(probabilities))
        mean_logit = float(np.mean(logits))
        band = (
            "Likely synthetic"
            if mean_logit >= self.band_upper
            else "Likely genuine"
            if mean_logit <= self.band_lower
            else "Inconclusive"
        )
        return {
            "score": score,
            "band": band,
            "per_window_scores": probabilities,
            "duration_seconds": clip.original_duration_s,
            "warnings": self._warnings(clip.waveform.numpy(), clip.sample_rate),
            "waveform": self._downsample(clip.waveform.numpy()),
            "limitations": LIMITATIONS,
        }

    def _warnings(self, waveform: np.ndarray, sample_rate: int) -> list[str]:
        warnings = []
        duration = len(waveform) / sample_rate
        peak, rms = float(np.max(np.abs(waveform))), float(np.sqrt(np.mean(waveform**2)))
        if duration < 2:
            warnings.append("Clip is shorter than 2 seconds; interpretation may be less reliable.")
        if rms < 0.01:
            warnings.append("Very low audio level detected.")
        if peak > 0.98:
            warnings.append("Possible clipping detected.")
        if rms > 1e-6:
            spectrum = np.abs(np.fft.rfft(waveform))
            frequencies = np.fft.rfftfreq(len(waveform), 1 / sample_rate)
            high_ratio = float(np.sum(spectrum[frequencies > 4000]) / max(np.sum(spectrum), 1e-8))
            if high_ratio < 0.02:
                warnings.append("Possible telephone-band-limited audio detected.")
            if float(np.std(waveform - np.convolve(waveform, np.ones(9) / 9, mode="same"))) > 0.2:
                warnings.append("Heavy noise may affect interpretation.")
        warnings.append(
            "Language not verified (the model was trained and tested on Hindi, Tamil "
            "and Bengali only)."
        )
        return warnings

    @staticmethod
    def _downsample(waveform: np.ndarray, points: int = 800) -> list[float]:
        indices = np.linspace(0, len(waveform) - 1, min(points, len(waveform))).astype(int)
        return waveform[indices].astype(float).tolist()


def create_app(model_path: str | Path | None = None) -> FastAPI:
    service = InferenceService(model_path=model_path)
    app = FastAPI(title="InDivoice research demo", version="0.8.0")
    app.state.service = service

    @app.get("/", response_class=HTMLResponse)
    def root() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_available": service.available,
            "model_path": str(service.model_path),
            "model_error": service.model_error,
        }

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)) -> dict:  # noqa: B008
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Unsupported audio format; accepted formats are WAV, MP3, FLAC, OGG, "
                    "M4A, and OPUS"
                ),
            )
        content = await file.read(service.max_bytes + 1)
        return service.analyze(content, suffix)

    return app


app = create_app()
