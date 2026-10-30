# Short paper or project-report outline

This outline is a production plan, not a claim that the results already exist.

## 1. Abstract

State the research question, languages, main B1 model, evaluation protocols, and the most important generalization result only after the corresponding result JSON exists. If results are absent, write a systems-and-protocol abstract and say so.

## 2. Introduction

Motivate multilingual synthetic-speech detection and explain why a positive model score is not evidence of fraud or trustworthiness. State the five research questions from [`docs/EXPERIMENTS.md`](EXPERIMENTS.md).

## 3. Related work

Discuss IndicSynth, IndicSUPERB/Kathbath, Indic-CodecFake, SEA-Spoof, MLAAD, and multilingual anti-spoofing. Feed citations from [`CITATION.cff`](../CITATION.cff) and [`docs/THIRD_PARTY.md`](THIRD_PARTY.md).

## 4. Data and leakage controls

Describe dataset access and licenses using `results/data_audit.json` and `docs/DATASETS.md`. Explain the IndicSynth/IndicSUPERB/Indic-CodecFake corpus-confound risk. Exact table: **Table 1 — dataset inventory, languages, licenses, roles, access status, and verified date**, fed by `docs/DATASETS.md` and `results/data_audit.json`.

## 5. Method

Describe preprocessing, log-Mel/LFCC features, B0–B3 models, training, calibration, fixed development threshold, window aggregation, clustered bootstrap, and robustness transformations. Exact figure: **Figure 1 — preprocessing and evaluation pipeline**, generated from the method description. Exact table: **Table 2 — model variants and parameter counts**, fed by export metadata and `results/benchmark.json` after real exports exist.

## 6. Experimental protocols

Define P1 in-distribution, P2 leave-one-generator-out, P3 leave-one-language-out, P4 pairwise transfer, P5 independent external genuine evaluation, and ablations. Exact table: **Table 3 — protocol/configuration matrix**, fed by `docs/EXPERIMENTS.md`.

## 7. Results

Include only real run artifacts. Exact table: **Table 4 — main metrics by protocol and model**, fed by evaluation JSON/CSV under `results/`. Exact table: **Table 5 — fixed-dev-threshold versus EER-threshold FPR/FNR**, fed by the same evaluation JSON. Exact figure: **Figure 2 — ROC and reliability diagrams**, fed by evaluator PNGs and ECE fields. Exact figure: **Figure 3 — P3/P4 language-transfer heatmaps**, fed by evaluator heatmaps.

## 8. Robustness and efficiency

Exact table: **Table 6 — robustness AUC/EER/FPR/FNR by degradation and level**, fed by `results/robustness/*.json` and CSV. Exact figure: **Figure 4 — EER versus degradation level with augmentation on/off**, fed by robustness plots. Exact table: **Table 7 — ONNX size, parameter count, p50/p95 latency, real-time factor, and memory**, fed by `results/benchmark.json` and `results/benchmark.md`.

## 9. Limitations and ethics

Use [`docs/LIMITATIONS.md`](LIMITATIONS.md) without weakening its claims. Discuss no scam-call validation, non-commercial data constraints, demographic bias, language limits, calibration shift, and generator obsolescence.

## 10. Reproducibility and conclusion

Point to the tagged repository, exact configs, manifests, resolved run configs, and commands in the README. Do not present fixture-test results as scientific results. Conclude with the narrow claim supported by the available protocol and uncertainty interval.
