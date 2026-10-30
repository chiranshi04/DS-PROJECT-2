"""Export B1 with ``python -m indivoice.export``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .exporter import export_b1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", choices=["tiny", "small"], default="small")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fixed-threshold", type=float, default=0.0)
    parser.add_argument("--eer-threshold", type=float)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--dev-npz", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}; no export was created.")
    result = export_b1(
        args.checkpoint,
        args.output,
        args.variant,
        temperature=args.temperature,
        fixed_threshold=args.fixed_threshold,
        eer_threshold=args.eer_threshold,
        opset=args.opset,
        quantize=args.quantize,
        dev_npz=args.dev_npz,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
