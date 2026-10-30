# Methodology

## Data sources and licensing

The project’s intended core synthetic source is IndicSynth, a multilingual synthetic-speech dataset covering 12 low-resource Indian languages and reporting approximately 4,000 hours from 989 target speakers. Its authors release it under CC BY-NC 4.0, so it is not a commercial-use dataset without separate permission [1]. IndicSUPERB/Kathbath is the principal genuine-speech source considered for the core protocols. Its official repository describes 1,684 hours across 12 Indian languages and states that the data packaging is CC0, while also warning that the underlying IndicCorp text is not owned by AI4Bharat; this distinction must be retained in any redistribution decision [2].

Indic-CodecFake is an external codec-synthesis benchmark constructed from IndicSUPERB real speech and neural audio codecs [3]. The repository/code and benchmark audio have different provenance questions; the dataset-specific audio license is not treated as verified in this project. P5 external genuine-speech checks use the candidate corpora listed in `configs/data/protocols/P5.yaml`: SEA-Spoof, Indic-CodecFake, Common Voice, FLEURS, and IndicVoices. Their access gates and licenses are recorded in [`docs/DATASETS.md`](DATASETS.md). SEA-Spoof is a published multilingual dataset spanning Tamil, Hindi, Thai, Indonesian, Malay, and Vietnamese [4]. MLAAD is cited as a relevant external anti-spoofing dataset but is not currently selected in P5; its paper reports synthetic audio from 205 TTS models in 54 languages [5].

No dataset is downloaded by this repository. A user must verify current terms, access conditions, speaker/privacy rights, and redistribution restrictions before use. The machine-readable audit is [`results/data_audit.json`](../results/data_audit.json).

## Split design

P1 is the in-distribution baseline. Training, development, and test assignments are speaker-disjoint, with class balancing controlled by `configs/data.yaml`. P2 holds out generators: a generator present in the test condition is not available to the training condition. P3 holds out a language. P4 measures pairwise language transfer among Hindi, Tamil, and Bengali. P5 is test-only external evaluation and is not used to train, tune, or calibrate the main model.

Every manifest row is expected to carry an utterance identifier, path, label, speaker, generator, language, corpus, and content/source identifier. The manifest utilities check required columns, duplicate hashes, speaker overlap, generator overlap where requested, and source-utterance overlap. Random selection is seeded and deterministic. Multi-window crops from one utterance stay in the same split.

## Corpus-confound risk

IndicSynth’s bona fide references are connected to the IndicSUPERB/Kathbath ecosystem, and Indic-CodecFake explicitly uses IndicSUPERB as its real corpus [2] [3]. If a detector trains on IndicSUPERB-derived real speech and is then tested on IndicSynth or Indic-CodecFake, it may exploit speaker, transcript, corpus, file-format, or recording-chain artifacts rather than synthetic-speech properties. The Phase 2 audit found no basis to assume that the relationship is harmless or that file-level duplication is absent.

The mitigation is to compare identifiers and hashes before manifest construction, enforce speaker/source-utterance disjointness, keep P2/P3/P4 separate from P1, and use independent genuine corpora in P5. Independent corpora do not eliminate all domain shift or licensing concerns, but they make it harder for a corpus-specific real-side shortcut to masquerade as detection ability. Any unresolved overlap is reported rather than silently removed.

## Preprocessing and features

Audio is decoded from common formats, converted to mono, resampled to 16 kHz, peak-normalized, and represented as fixed four-second windows. Training uses seeded random crops. Testing uses deterministic 50%-overlap multi-window coverage with zero padding or tiling for short clips. Silence trimming is off by default and is an explicit ablation because silence can be a shortcut. The main feature is an 80-bin log-Mel spectrogram with a 25 ms window, 10 ms hop, 512-point FFT, epsilon-protected log compression, and per-utterance mean/variance normalization. LFCC is used for the classical B0 baseline.

## Models and training

B0 is the LFCC plus class-conditional GMM baseline. B1 is a small log-Mel CNN with tiny and small variants, global pooling, and one logit. B2 is a compact raw-waveform reference. B3 is an optional frozen pretrained SSL reference and is disabled by default. B1 training uses AdamW, class-weighted BCE-with-logits, a cosine schedule, gradient clipping, early stopping on development EER, checkpointing, resolved-config saving, and deterministic seeds. No result is valid unless the run directory contains the source configuration and the data/manifests used to produce it.

## Metrics and threshold policy

ROC-AUC summarizes ranking quality. EER is the operating point where false-accept and false-reject rates intersect; the corresponding threshold is reported. FPR and FNR are reported twice: at a threshold fixed on the training-condition development set and at the EER threshold for comparison. The difference between those operating points makes out-of-domain threshold shift visible. minDCF is reported when the required costs and priors are available. ECE uses reliability bins and is accompanied by a reliability diagram.

Window scores are aggregated by the configured mean of logits; median is available as an explicit alternative. Calibration uses temperature scaling fit on development data only and saves the fitted temperature with the run/export metadata. P5 external corpora never tune this temperature or the operating threshold.

## Confidence intervals

The default is a 95% speaker-clustered bootstrap with 1,000 resamples. A bootstrap draw samples speakers with replacement and includes all utterances/windows belonging to each sampled speaker. It must not resample individual files as if they were independent observations. The reported interval is therefore conditional on the observed speaker population and should not be interpreted as a population guarantee beyond the sampled corpora.

## Reproducibility record

Use [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) for the question-to-command map. Keep manifests, resolved configs, seeds, model checkpoints, prediction files, metric JSON/CSV, and generated figures together. The report builder refuses missing inputs, and the README explicitly excludes fixture measurements from scientific results.

## References

[1]: https://aclanthology.org/2025.acl-long.1070/ "IndicSynth: A Large-Scale Multilingual Synthetic Speech Dataset for Low-Resource Indian Languages"
[2]: https://github.com/AI4Bharat/indicSUPERB "AI4Bharat IndicSUPERB repository and Kathbath documentation"
[3]: https://aclanthology.org/2026.findings-acl.2159/ "Indic-CodecFake meets SATYAM"
[4]: https://arxiv.org/html/2509.19865v2 "SEA-Spoof: Bridging the Gap in Multilingual Audio Deepfake Detection for South-East Asia"
[5]: https://arxiv.org/abs/2401.09512 "MLAAD: The Multi-Language Audio Anti-Spoofing Dataset"
