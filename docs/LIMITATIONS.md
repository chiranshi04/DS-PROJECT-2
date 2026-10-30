# Limitations

## Evidence and deployment scope

This repository does not contain real scam-call data. Results on curated read speech, synthetic speech, or benchmark corpora cannot establish performance on spontaneous calls, adversarial callers, channel mixtures, consent conditions, or real fraud investigations. The web application is a local research demo and must not be used as the sole basis for a decision.

## Data rights and access

The data plan includes CC BY-NC material, gated datasets, and sources whose raw text, audio, generated derivatives, or redistribution terms require separate review. A citation is not a license. Users must satisfy the current terms of every source and must not redistribute downloaded audio or generated derivatives unless the applicable terms allow it.

## Generators and domain shift

The generators named in the available corpora are research or documented benchmark systems. They may not represent current commercial voice cloners, private APIs, watermarking schemes, multilingual fine-tunes, or deliberate post-processing. Codec, noise, reverberation, telephone bandwidth, compression, and adversarial editing can change detector behavior.

## Languages and calibration

The project’s primary app language warning names Hindi, Tamil, and Bengali. Broader candidate corpora contain additional languages, but broader coverage does not mean validated performance. Calibration temperature and decision thresholds are valid only for the training condition and development procedure that produced them. Out-of-domain language, generator, corpus, speaker, or channel shift can invalidate both calibration and outcome bands.

## Bias and representation

Speaker counts, gender balance, recording environments, accents, dialects, age, accessibility, and socioeconomic coverage may be uneven. A detector can encode demographic or corpus shortcuts even when aggregate AUC appears strong. Per-language and per-group breakdowns are required, but small groups make uncertainty large and do not prove absence of bias.

## Missing results

No real-data model-performance, robustness, calibration, or deployment metrics are currently committed under `results/`. Fixture tests verify software behavior only. Any future result must include its source data, protocol, configuration, predictions, threshold policy, confidence interval procedure, and license/provenance record.
