"""Save log-Mel example plots from a local manifest only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from indivoice.data.audio import load_audio, make_windows
from indivoice.data.manifests import read_manifest
from indivoice.features.mel import LogMelExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/examples"))
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = LogMelExtractor()
    counts = {0: 0, 1: 0}
    for record in read_manifest(args.manifest):
        if counts.get(record.label, 0) >= args.per_class:
            continue
        clip = load_audio(record.path_or_uri)
        window = make_windows(clip.waveform, clip.sample_rate, training=False)[0]
        feature = extractor(window.waveform).numpy()
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.imshow(feature, origin="lower", aspect="auto", interpolation="nearest")
        axis.set(
            title=f"label={record.label} language={record.language}",
            xlabel="frame",
            ylabel="Mel bin",
        )
        figure.tight_layout()
        output = args.output_dir / f"label-{record.label}-{counts[record.label]}.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        counts[record.label] += 1
        if all(value >= args.per_class for value in counts.values()):
            break
    print(f"Saved examples under {args.output_dir}; observed counts={counts}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
