# Audio preprocessing and feature shapes

All audio is decoded to mono, resampled to **16,000 Hz**, peak-normalized, and represented as a fixed **4.0-second** window containing 64,000 samples. A shorter clip is zero-padded by default; deterministic tiling is available as an explicit alternative. Silence trimming is intentionally not applied by default because silence can become a shortcut in spoof datasets.

## Windowing

Training returns one reproducible random crop per manifest row when given a seed. Evaluation returns deterministic windows with 50% overlap. A 4-second clip with the default centered STFT settings produces 401 frames. The original clip duration is retained in each dataset item.

## Log-Mel

The default log-Mel extractor uses `n_fft=512`, a 25 ms (`win_length=400`) Hann window, a 10 ms (`hop_length=160`) hop, 80 Mel bins, and a 20–7,600 Hz frequency range. It returns a tensor shaped `(80, 401)` for a 4-second crop. Power is clamped by epsilon before natural-log compression, then normalized per utterance to mean approximately zero and standard deviation approximately one.

## LFCC

The classical baseline returns a normalized linear-frequency cepstral representation shaped `(40, 401)` with the default 40 coefficients. The same waveform/windowing and normalization rules apply.

Feature caches are optional, stored under a caller-provided cache directory, and excluded from Git. On-the-fly augmentation is accepted as a callable hook but is intentionally not populated until the robustness phase.
