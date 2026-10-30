# InDivoice

InDivoice is a CPU-capable research system for studying detection of genuine versus synthetic speech in Hindi, Tamil, and Bengali. The project uses fixed-window audio preprocessing, 80-bin log-Mel features, a compact CNN main model, leakage-aware protocols, calibration, robustness transformations, ONNX export, and a small local FastAPI demo. It is a research implementation, not a forensic, authentication, fraud, or trustworthiness system.

## Limitations and intended use

Synthetic speech does **not** imply fraud, malicious intent, or wrongdoing. Genuine speech is **not necessarily trustworthy**. A model result can be wrong and must not be used as the sole basis for financial, legal, employment, safety, identity, access, surveillance, or moderation decisions. The current data plan does not contain real scam-call data, includes non-commercial or gated sources, and may not represent current commercial voice cloners. Language, demographic, speaker, corpus, microphone, codec, and generator shift can materially change performance. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before using any result.

The local web demo processes uploads in memory or in a request-local temporary file deleted after decoding. It does not log audio bytes, file names, or predictions. It never fabricates a prediction when the configured ONNX model is absent.

## Installation

Python 3.10 or newer is supported. For a clean CPU environment:

```bash
git clone https://github.com/chiranshi04/DS-PROJECT-2.git
cd DS-PROJECT-2
make install
make test
make lint
```

The default dependency set is CPU-compatible. A GPU is optional for acceleration, not required for the fixture tests or the documented CPU path.

## Quick start

The repository contains metadata audit evidence and implementation tests, but no downloaded speech corpus, trained checkpoint, or real-data performance result. To run the local app after obtaining a verified B1 checkpoint and exporting it:

```bash
python -m indivoice.export --checkpoint runs/p1_b1_small/best.pt --output releases/b1.onnx --variant small
make app
# Open http://127.0.0.1:8000
```

If `releases/b1.onnx` is missing, `/health` reports `model_available: false`, the page shows a model-not-found state, and `/analyze` returns an error instead of producing a fake result.

## Reproduce the experiments from scratch

The sequence below is intentionally gated by local data availability. Do not skip the provenance and manifest steps. The commands that require data fail clearly when manifests, audio, checkpoints, or prediction files are absent.

### 1. Audit sources and build manifests

```bash
make audit-data
make build-manifests
```

`make audit-data` writes the metadata-first audit to `results/data_audit.json`. Place downloaded data under the configured paths in `configs/data.yaml`, follow the access and license notes in [`docs/DATASETS.md`](docs/DATASETS.md), then build the P1–P5 manifests:

```bash
python scripts/build_manifests.py --source data/manifests/source_records.csv --output-dir data/manifests
```

The manifest builder enforces required metadata, speaker/source-utterance overlap checks, deterministic subset selection, and protocol assignment. It does not download data or waive a dataset license.

### 2. Train the requested configurations

```bash
make train EXP=p1_b1_small
make train EXP=p1_b0_gmm
make train EXP=p2_b1_logo
make train EXP=p3_b1_lolo
make train EXP=p4_pairwise
make train EXP=silence_trim_on_off
make train EXP=codec_aug_on_off
make train EXP=model_size
```

The B0 GMM configuration is recorded for the classical baseline, while the current training module requires a later dedicated B0 fitting path. A command that cannot find verified manifests stops without claiming a result. Expected run artifacts are written below each configured `runs/` directory, including checkpoints, resolved config, CSV logs, and best/last state where training completes.

### 3. Evaluate with dev-only thresholds and calibration

```bash
python -m indivoice.eval --run runs/p1_b1_small --protocol P1 --split test
python -m indivoice.eval --run runs/p2_b1_logo --protocol P2 --split test
python -m indivoice.eval --run runs/p3_b1_lolo --protocol P3 --split test
python -m indivoice.eval --run runs/p4_pairwise --protocol P4 --split test
python -m indivoice.eval --run runs/p5_external --protocol P5 --split test
python scripts/make_report_tables.py --results-dir results --output results/evaluation_tables.md
```

The evaluator fixes the operating threshold using the training-condition development set, fits temperature scaling on development data only, aggregates multi-window logits by the configured mean, and writes confidence intervals using speaker-clustered bootstrap. The report builder refuses missing result JSON files rather than creating placeholders.

### 4. Run robustness experiments

After producing a scored test CSV with `degradation`, `level`, `label`, and `logit` columns, run:

```bash
make robustness SCORES=results/robustness_scores.csv THRESHOLD=0.0 OUT=results/robustness
```

The default degradation plan is in [`configs/robustness.yaml`](configs/robustness.yaml). It includes codec levels, synthetic colored noise, simulated reverberation, 8 kHz round trips, low-pass filtering, duration sweeps, and seeded training-time augmentation. Synthetic colored noise is used unless separately licensed MUSAN or DEMAND recordings are supplied.

### 5. Export and benchmark

```bash
python -m indivoice.export --checkpoint runs/p1_b1_small/best.pt --output releases/b1.onnx --variant small
python scripts/benchmark.py --model-dir releases --output-json results/benchmark.json --output-md results/benchmark.md --runs 100 --warmup 10
```

Export writes a JSON sidecar containing feature settings, calibration temperature, thresholds, parameter count, and parity tolerances. The benchmark reports actual single-thread and default-thread CPU measurements only for ONNX files that exist. Optional dynamic INT8 comparison requires a development `.npz` with `features` and `labels`:

```bash
python -m indivoice.export --checkpoint runs/p1_b1_small/best.pt --output releases/b1.onnx --variant small --quantize --dev-npz results/dev_features.npz
```

### 6. Run the web demo

```bash
make app
# or
 docker build -t indivoice-demo .
 docker run --rm -p 8000:8000 -v "$PWD/releases:/app/releases:ro" indivoice-demo
```

The app accepts WAV, MP3, FLAC, OGG, M4A, and OPUS files up to 20 MB and 60 seconds. Its result is a model confidence indication with an outcome band, not a claim of truth.

## Expected hardware and runtime

Fixture tests and lint are designed for an ordinary CPU machine with at least 4 GB RAM and several hundred MB of free disk. Metadata audit and manifest construction need enough disk for the selected manifests. Real training and evaluation require storage for the downloaded corpora, generated feature/cache files, checkpoints, and result files; the required amount depends on the selected subset and should be measured before downloading. A GPU with 8 GB or more VRAM can accelerate training, but the documented configurations default to CPU and the B1 model is intentionally lightweight. Full multilingual, cross-generator, and robustness experiments can take from hours to days depending on corpus size, storage, CPU/GPU, and number of bootstrap resamples.

## Results policy

No real experiment result files are currently present under `results/`. The tracked `results/data_audit.json` is provenance evidence, not a model-performance result. Therefore this README does not present a results table. Run the commands above after a licensed data download and full experiment to generate JSON, CSV, Markdown, and plot artifacts. Do not copy fixture-test measurements into a scientific results table.

## Project layout

- `configs/` — versioned data, feature, model, experiment, evaluation, robustness, export, and app configuration.
- `src/indivoice/` — package code for data, features, models, training, evaluation, robustness, and export.
- `scripts/` — audit, manifest, visualization, report, benchmark, and compatibility entry points.
- `app/` — FastAPI backend and plain HTML/CSS/JavaScript frontend.
- `tests/` — CPU fixture and HTTP tests.
- `docs/` — methodology, experiment mapping, limitations, paper outline, provenance, and runbook.
- `results/` — only real recorded artifacts and the committed metadata audit.

## Contributing and verification

Before opening a pull request, run:

```bash
make format
make lint
make test
git diff --check
```

Keep audio, checkpoints, caches, generated plots, and unverified result files out of Git. Add or update tests for behavior changes. Do not report metrics unless the source JSON/CSV and the exact configuration exist. Keep license and provenance changes in [`docs/DATASETS.md`](docs/DATASETS.md) and update [`CITATION.cff`](CITATION.cff) or [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) when external work is added.
