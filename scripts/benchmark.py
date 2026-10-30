"""Benchmark existing ONNX models without inventing absent measurements."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _peak_memory_mb() -> float:
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
    except (ImportError, AttributeError):
        return float("nan")


def benchmark_model(
    path: Path, runs: int = 100, warmup: int = 10, thread_mode: str = "default"
) -> dict:
    """Measure one model after warm-up; thread_mode is single or default."""
    import onnxruntime as ort

    options = None
    if thread_mode == "single":
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    input_shape = [
        1 if isinstance(dim, str) or dim is None else int(dim)
        for dim in session.get_inputs()[0].shape
    ]
    input_shape = [1, 80, 401] if len(input_shape) != 3 else input_shape
    features = np.zeros(input_shape, dtype=np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: features})
    start_memory = _peak_memory_mb()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: features})
        timings.append(time.perf_counter() - start)
    latency_p50, latency_p95 = np.percentile(timings, [50, 95])
    parameters = None
    metadata = path.with_suffix(".json")
    if metadata.exists():
        parameters = json.loads(metadata.read_text(encoding="utf-8")).get("parameters")
    return {
        "model": str(path),
        "thread_mode": thread_mode,
        "parameters": parameters,
        "onnx_size_mb": path.stat().st_size / 1_000_000,
        "latency_p50_ms": float(latency_p50 * 1000),
        "latency_p95_ms": float(latency_p95 * 1000),
        "real_time_factor_p50": float(4.0 / latency_p50),
        "peak_memory_mb": _peak_memory_mb() - start_memory,
        "runs": runs,
        "warmup": warmup,
        "threads": 1 if thread_mode == "single" else os.cpu_count(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("releases"))
    parser.add_argument("--output-json", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/benchmark.md"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    models = sorted(args.model_dir.glob("*.onnx"))
    if not models:
        raise SystemExit(
            f"No ONNX models found in {args.model_dir}; refusing to invent benchmark numbers."
        )
    records = [
        benchmark_model(model, args.runs, args.warmup, mode)
        for model in models
        for mode in ("single", "default")
    ]
    payload = {"cpu_model": _cpu_model(), "os": platform.platform(), "records": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    headers = [
        "model",
        "thread_mode",
        "parameters",
        "onnx_size_mb",
        "latency_p50_ms",
        "latency_p95_ms",
        "real_time_factor_p50",
        "peak_memory_mb",
    ]
    lines = [
        f"# Efficiency benchmark\n\nCPU: `{payload['cpu_model']}`\n\nOS: `{payload['os']}`\n",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for record in records:
        lines.append("| " + " | ".join(str(record[key]) for key in headers) + " |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
