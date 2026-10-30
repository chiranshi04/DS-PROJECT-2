# InDivoice B1 model card

## Intended use

B1 is a research-only binary classifier for studying whether speech samples in controlled datasets are labeled genuine or synthetic. It is intended for reproducible experiments, ablations, and robustness analysis—not for identity, employment, legal, safety, moderation, or fraud decisions.

## Training data

The model is designed for the licensed datasets and leakage-safe protocols recorded in [`docs/DATASETS.md`](../docs/DATASETS.md). Actual B1 training on real project data has not yet been run in this repository. Data licenses, language coverage, and manifest provenance must be verified before training.

## Languages

Not yet run. The intended language coverage is determined by the verified dataset manifests and protocol configuration, not by this model card.

## Metrics

Real-data training, calibration, test evaluation, robustness, and efficiency metrics: **not yet run**. Fixture tests verify implementation behavior only and are not model-performance results.

## Known failure modes

Performance may depend on silence, recording channel, codec, sampling rate, speaker overlap, generator artifacts, language, and corpus-specific shortcuts. Thresholds can shift out of domain. Synthetic speech does not imply fraud, and genuine speech is not necessarily trustworthy. The model can fail on unseen generators, languages, microphones, codecs, noise, reverberation, and short clips.

## Limitations and intended use

This is a research demo only. It must not be used as proof that a speaker, recording, or claim is fraudulent or trustworthy. It does not establish provenance, intent, identity, consent, or truthfulness. Training and evaluation data must remain within their stated licenses; current project data documentation includes CC BY-NC restrictions where applicable.

## Export

ONNX exports include a JSON sidecar containing feature extraction settings, calibration temperature, decision thresholds, input shape, and parity tolerance. ONNX Runtime parity is asserted during export. Dynamic INT8 quantization is optional and must be evaluated on a development set before any comparison is interpreted.
