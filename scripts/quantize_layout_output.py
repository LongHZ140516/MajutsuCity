#!/usr/bin/env python3
"""Call the upstream palette quantizer with an exact destination filename."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_ROOT = PROJECT_ROOT / "layout_gen"
sys.path.insert(0, str(LAYOUT_ROOT))

from quantize_palette import process_one  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize one layout with the upstream helper and save it exactly."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    source = args.input_file.expanduser().resolve()
    destination = args.output_file.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Layout image not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    process_one(source, destination, verbose=args.verbose)
    print(f"Saved quantized layout: {destination}", flush=True)


if __name__ == "__main__":
    main()
