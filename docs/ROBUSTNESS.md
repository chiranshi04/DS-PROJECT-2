# Robustness experiments

All degradations accept a mono NumPy waveform and an explicit sample rate. The functions are deterministic when their seed is fixed. They return audio at the input sample rate unless a function is explicitly documented as a helper with a different intermediate rate.

## Codecs

`codec_degrade` writes a temporary WAV, invokes ffmpeg, and decodes the result back to a mono waveform at the original rate. The configured levels are MP3 at 128/64/32 kbps, AAC at 64 kbps, Opus at 24/12 kbps, and G.711 mu-law after an 8 kHz telephone-band round trip. Temporary files are deleted. Tests skip codec cases with a clear reason when ffmpeg is unavailable.

## Noise

The default noise source is deterministic colored noise, not MUSAN or DEMAND. This avoids silently downloading or redistributing external recordings. White, pink, and brown noise are supported. Noise is scaled to SNR 20, 10, 5, or 0 dB using the clean-signal and noise powers. If MUSAN or DEMAND is used later, its source version and license must be recorded here before results are compared.

## Reverberation and bandwidth

Reverberation convolves the waveform with a seeded exponentially decaying simulated room impulse response with configurable RT60. Resampling performs a 16 kHz → 8 kHz → 16 kHz round trip. Low-pass filtering uses a fourth-order Butterworth filter, with configured cutoffs of 3,500 and 2,500 Hz.

## Duration and augmentation

The duration sweep crops or zero-pads to 1, 2, 3, and 4 seconds. `RandomAugment` samples one configured operation from a seeded generator for training-time augmentation. Set `augmentation.enabled` to compare augmentation-on and augmentation-off runs with the same manifest, seed, optimizer, and evaluation threshold.

## Scoring

The robustness runner consumes a scored CSV with `degradation`, `level`, `label`, and `logit` columns. The fixed threshold must come from the training-condition development set; it is never tuned on degraded test data. The runner writes JSON, CSV, and EER-versus-level PNG outputs. Use:

```bash
make robustness SCORES=/path/to/scored_degradations.csv THRESHOLD=0.42
```

No placeholder result is created when the scores file is missing. All real experiments must record the model run, source manifest, degradation seed, exact configuration, and fixed threshold.
