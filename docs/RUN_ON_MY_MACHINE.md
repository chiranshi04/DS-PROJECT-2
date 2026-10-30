# Run Phase 2 data work locally

The committed audit records what was verified from official source pages and what remained inaccessible. No audio or gated content is included in this repository.

## Install and probe official metadata

```bash
make install
python scripts/audit_data.py --output results/data_audit.json
```

The audit uses the configured official URLs in `configs/data_sources.yaml`. Hugging Face repositories are inspected through `huggingface_hub`. Public/no-gate metadata can be listed without credentials; gated repositories require the user to authenticate and accept the dataset's own conditions before content becomes available.

## Prepare a verified source manifest

After downloading data under the source's license and inspecting its metadata, create `data/manifests/source_records.csv` with exactly these columns:

```text
path_or_uri,sha256,label,language,speaker_id,source_corpus,generator_id,duration_s,sample_rate,split
```

Use label `0` for genuine human speech and `1` for synthetic speech. Use stable speaker IDs and generator IDs from the source metadata. Compute SHA-256 only for local files; keep a blank value for remote URIs until a local copy exists. Do not assign a split until speaker and generator policies are explicit.

## Build and validate protocols

```bash
make build-manifests
pytest
```

The builder writes P1, P2 per-generator/grouped, P3 per-language, P4 pairwise-language, and P5 external-test-only CSVs under `data/manifests/`. It applies the configurable duration cap in `configs/data.yaml` and fails on speaker, held-out-generator, held-out-language, or duplicate-hash leakage.

Before using a candidate independent genuine corpus, create a comparable manifest and compare it with the IndicSynth bona fide/reference manifest using `compare_metadata_overlap`. A zero result is evidence only for the fields and versions actually compared; it is not a general proof of independence.

Do not commit audio, datasets, credentials, gated content, or checkpoints. Record the exact source version, download command, license acceptance, manifest hash, and command output in the experiment record.

## Phase 4 model training

After creating verified protocol manifests and installing dependencies:

```bash
make train EXP=p1_b1_small
python -m indivoice.train --config configs/experiments/p1_b1_small.yaml
```

The CLI requires real `train_manifest` and `dev_manifest` paths and refuses to start when they are missing. It writes the resolved configuration, CSV metrics, `best.pt`, and `last.pt` under the configured run directory. Training uses AdamW, cosine learning-rate decay, weighted BCE-with-logits, gradient clipping, early stopping on development EER, and CUDA mixed precision when enabled and available. B3 is intentionally off by default and requires an additional Transformers environment and substantial disk/GPU memory.

For the lightweight B1 model, expect approximately 0.1–1 MB of model weights, under 5 MB for an ONNX export, and CPU training time that depends on the manifest size and number of epochs. A small fixture run completes in seconds; a real multi-hour corpus may require hours on CPU. Feature caches can approach the size of the decoded dataset and should be placed on a persistent disk outside Git. GPU training is optional for B1/B2; 8 GB VRAM is a practical starting point for larger batches, while the B3 reference may require substantially more.

A Colab-friendly sequence is:

```bash
!git clone https://github.com/chiranshi04/DS-PROJECT-2.git
%cd DS-PROJECT-2
!pip install -r requirements-dev.txt
!pip install -e .
!python scripts/audit_data.py --output /content/data_audit.json
# After licensed data download and manifest creation:
!python scripts/build_manifests.py
!python -m indivoice.train --config configs/experiments/p1_b1_small.yaml
!pytest
```

No Phase 4 real-data training result is reported by this repository until the exact dataset version, manifest, configuration, and command are recorded.

## Phase 5 evaluation

After a real training run has produced prediction CSVs for both development and test data, run:

```bash
python -m indivoice.eval --run runs/p1_b1_small --protocol P1 --split test
make evaluate RUN=runs/p1_b1_small PROTOCOL=P1 SPLIT=test
python scripts/make_report_tables.py --results-dir results
```

The runner requires `predictions_<protocol>_dev.csv` and `predictions_<protocol>_<split>.csv`; it refuses to create placeholders when either is absent. Each prediction CSV must include `logit`, `label`, `speaker_id`, `language`, and `source_corpus`, with optional `utterance_id`, `generator_id`, and `gender`. Mean or median window logits are aggregated by utterance before scoring. Temperature scaling and the fixed operating threshold are fitted on development predictions only and saved as `temperature.json`. Test EER is reported with its own threshold only as a comparison; it is never used as the fixed deployment threshold.

Evaluation writes JSON and CSV results under `results/`, plus language-transfer heatmaps for P3/P4. Confidence intervals resample speakers, not files, with 1,000 bootstrap resamples by default. The report builder refuses to run if actual evaluation JSONs are missing.

## Phase 8 web demo

Place a verified B1 ONNX export at `releases/b1.onnx` with its JSON sidecar, then run:

```bash
make app
# open http://127.0.0.1:8000
# or: docker build -t indivoice-demo . && docker run --rm -p 8000:8000 -v "$PWD/releases:/app/releases:ro" indivoice-demo
```

The app accepts WAV, MP3, FLAC, OGG, M4A, and OPUS files up to 20 MB and 60 seconds. It decodes through the same preprocessing path, uses deterministic multi-window inference, applies the exported temperature and threshold metadata, and shows warnings plus the fixed limitations panel. If the configured model is absent, `/health` reports `model_available: false`, the page shows “Model not found,” and `/analyze` returns HTTP 503; no prediction is fabricated.

Uploads are read into memory or a request-local temporary file that is deleted immediately after decoding. Audio bytes, file names, and predictions are not logged by the application. This is a local research demo; do not expose it publicly without adding authentication, rate limits, encrypted transport, and a reviewed data-retention policy.
