# Design decisions

## D001 — Python package under `src/`

Use a `src/indivoice` layout to prevent accidental imports from the repository root and to make editable installation explicit.

## D002 — One YAML file per concern

Keep data, features, model, experiment, evaluation, robustness, export, and app settings separate. This makes experiment inputs reviewable and avoids hidden defaults.

## D003 — OmegaConf as the configuration loader

OmegaConf provides typed-enough hierarchical access and clear missing-file errors while keeping configuration files ordinary YAML.

## D004 — Deterministic seed utility

Seed Python, NumPy, and Torch and enable deterministic Torch algorithms where available. Reproducibility is preferred over maximum throughput for the research baseline.

## D005 — No data or model artifacts in Git

Audio, datasets, checkpoints, and generated artifacts are excluded to protect licensing, repository size, and user privacy. Manifests and hashes will be committed instead.

## D006 — Infrastructure-only Phase 1

Do not fabricate dataset facts or results before the data audit and experiment code exist. Later phases must update `STATUS.md` with exact commands and verification limits.
