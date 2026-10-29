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
