#!/usr/bin/env python3
"""Generate a seamless water base-color texture with the MajutsuCity LoRA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_tiling import blend_wrapped_edges
from utils.qwen_lora_generation import generate_lora_image

DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen-Image"
DEFAULT_LORA = (
    PROJECT_ROOT
    / "models"
    / "MajutsuCity"
    / "texture"
    / "Texture_Lora.safetensors"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "texture_demo.png"
DEFAULT_PROMPT = (
    "A seamless, tileable PBR base-color texture of stylized water: crystal "
    "clear azure blue surface, soft painterly color variation, gentle ripple "
    "patterns, uniform lighting, flat top-down material scan, no perspective, "
    "no seams, no objects, no text."
)
DEFAULT_NEGATIVE = (
    "low quality, blurry, distorted, text, watermark, horizon, perspective, "
    "shore, coast, boat, object, frame, border, directional lighting"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one seamless water base-color texture using a local "
            "Qwen Image snapshot and the MajutsuCity texture LoRA."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record-path", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.steps <= 0:
        raise ValueError("width, height and steps must be positive")
    if args.true_cfg_scale <= 0:
        raise ValueError("--true-cfg-scale must be positive")

    resolved = {
        "kind": "water_basecolor",
        "model_path": str(args.model_path.expanduser().resolve()),
        "lora_path": str(args.lora_path.expanduser().resolve()),
        "output_path": str(args.output_path.expanduser().resolve()),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
    }
    print(json.dumps(resolved, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run completed; no model was loaded.", flush=True)
        return

    generate_lora_image(
        model_path=args.model_path,
        lora_path=args.lora_path,
        output_path=args.output_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
    )
    with Image.open(args.output_path) as image:
        blend_wrapped_edges(
            image, left_right=True, top_bottom=True
        ).save(args.output_path)
    resolved["postprocess"] = "blend_left_right_and_top_bottom_edges"
    if args.record_path is not None:
        record_path = args.record_path.expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("w", encoding="utf-8") as file:
            json.dump(resolved, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
