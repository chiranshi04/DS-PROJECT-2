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

## D007 — Retain Hindi, Tamil, and Bengali

The audit found all three languages explicitly represented across the principal resources used in this project. They remain the Phase 2 languages because this gives the clearest documented common intersection; it does not imply they are more representative or better licensed than other Indian languages.

## D008 — Treat dataset access and licensing as per-source properties

The audit found public, email-based, information-sharing, and manual-gated resources, plus an external resource without a verified bulk download. It also found license conflicts for Kathbath and an unspecified dataset/audio license for Indic-CodecFake. The project will not normalize these into a single license or silently treat gated data as available.

## D009 — Treat Indic-CodecFake and IndicSUPERB as overlapping by construction

The audit documents a one-to-one codec resynthesis relationship from IndicSUPERB utterances. They must not be treated as independent real/synthetic evidence without explicit protocol separation and file-level manifests.

## D010 — No overlap claim without comparable metadata

Speaker and source-utterance overlap counts are only reported when comparable manifests are available. Names, shared languages, citations, or shared text do not count as proof of duplicate speakers or files.
