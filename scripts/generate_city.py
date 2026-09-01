#!/usr/bin/env python3
"""Generate one styled MajutsuCity scene from a prompt or semantic layout."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import run_pipeline as pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PIPELINE = PROJECT_ROOT / "scripts" / "run_pipeline.py"


def available_styles() -> list[str]:
    return sorted(path.stem for path in (PROJECT_ROOT / "image_prompts").glob("*.json"))


def normalized_style(value: str) -> str:
    return value.strip().lower()


def parse_args() -> argparse.Namespace:
    defaults = pipeline.nested(
        pipeline.load_yaml(pipeline.DEFAULT_PIPELINE_CONFIG), "defaults"
    )
    styles = available_styles()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a styled city with one command. Supply a layout/depth pair, "
            "a layout only, or a text prompt; omitted inputs use the configured "
            "default layout prompt."
        )
    )
    parser.add_argument(
        "--style",
        type=normalized_style,
        choices=styles,
        help="City style. Names are case-insensitive.",
    )
    parser.add_argument(
        "--list-styles", action="store_true", help="List available styles and exit."
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Semantic layout image. A matching depth map is generated when --depth is omitted.",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=None,
        help="Existing height/depth image paired with --layout.",
    )
    parser.add_argument(
        "--layout-prompt",
        default=None,
        help="Prompt used to generate a layout when --layout is omitted.",
    )
    parser.add_argument(
        "--scene-prompt",
        default=None,
        help=(
            "High-level city description expanded through an OpenAI-compatible "
            "Scene Design API. The explicit --style remains a hard constraint."
        ),
    )
    parser.add_argument(
        "--scene-plan",
        type=Path,
        default=None,
        help="Reuse an existing validated Scene Plan JSON without calling an API.",
    )
    parser.add_argument(
        "--height-prompt",
        default=None,
        help="Optional prompt override when generating a height map.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Output case name. When omitted, a non-conflicting name is generated.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=int(defaults["seed"]))
    parser.add_argument("--stages", default=None)
    parser.add_argument("--building-ids", default=None)
    parser.add_argument(
        "--layout-palette", choices=("citydreamer", "majutsu", "auto"), default=None
    )
    parser.add_argument(
        "--edit-profile", choices=("legacy", "constrained"), default=None
    )
    parser.add_argument("--execution-order", choices=("phase", "style"), default=None)
    parser.add_argument(
        "--building-scale-mode", choices=("depth", "uniform"), default=None
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing-buildings", action="store_true")
    parser.add_argument(
        "--enable-vlm-review",
        action="store_true",
        help=(
            "Use an OpenAI-compatible vision model to review generated building, "
            "tree and lamp images before downstream 3D generation."
        ),
    )
    parser.add_argument(
        "--vlm-max-attempts",
        type=int,
        default=int(defaults.get("vlm_max_attempts", 3)),
        help="Maximum image-generation attempts when VLM review is enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print every stage without loading models or using GPUs.",
    )
    args = parser.parse_args()
    if args.list_styles:
        return args
    if args.style is None:
        parser.error("--style is required unless --list-styles is used.")
    if args.depth is not None and args.layout is None:
        parser.error("--depth requires --layout.")
    if args.layout is not None and args.layout_prompt is not None:
        parser.error("--layout cannot be combined with --layout-prompt.")
    if args.depth is not None and args.height_prompt is not None:
        parser.error("--height-prompt cannot be used when --depth is supplied.")
    if args.scene_prompt is not None and args.scene_plan is not None:
        parser.error("--scene-prompt and --scene-plan are mutually exclusive.")
    if args.vlm_max_attempts <= 0:
        parser.error("--vlm-max-attempts must be positive.")
    return args


def safe_case_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return normalized or "city"


def automatic_case_name(args: argparse.Namespace) -> str:
    if args.layout is not None:
        prefix = f"{safe_case_component(args.layout.stem)}_{args.style}"
    else:
        prefix = str(args.style)
    base = f"{prefix}_seed_{args.seed}"
    output_root = PROJECT_ROOT / "output"
    candidate = base
    suffix = 2
    while (output_root / candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def append_option(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def controller_command(args: argparse.Namespace) -> list[str]:
    case_name = args.case_name or automatic_case_name(args)
    command = [
        sys.executable,
        str(RUN_PIPELINE),
        "--style",
        str(args.style),
        "--case-name",
        case_name,
        "--seed",
        str(args.seed),
    ]
    append_option(command, "--config", args.config)
    append_option(command, "--scene-prompt", args.scene_prompt)
    append_option(command, "--scene-plan", args.scene_plan)
    append_option(command, "--height-prompt", args.height_prompt)
    append_option(command, "--stages", args.stages)
    append_option(command, "--building-ids", args.building_ids)
    append_option(command, "--layout-palette", args.layout_palette)
    append_option(command, "--edit-profile", args.edit_profile)
    append_option(command, "--execution-order", args.execution_order)
    append_option(command, "--building-scale-mode", args.building_scale_mode)

    if args.layout is not None:
        append_option(command, "--layout-path", args.layout)
        append_option(command, "--depth-path", args.depth)
    else:
        command.append("--generate-layout")
        append_option(command, "--layout-prompt", args.layout_prompt)

    if args.overwrite:
        command.append("--overwrite")
    if args.allow_missing_buildings:
        command.append("--allow-missing-buildings")
    if args.enable_vlm_review:
        command.extend(
            ["--enable-vlm-review", "--vlm-max-attempts", str(args.vlm_max_attempts)]
        )
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    args = parse_args()
    if args.list_styles:
        for style in available_styles():
            print(style)
        return

    command = controller_command(args)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
