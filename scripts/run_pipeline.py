#!/usr/bin/env python3
"""Run the open-source MajutsuCity pipeline from layout to Blender scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
DEFAULT_LOCAL_CONFIG = PROJECT_ROOT / "configs" / "paths.local.yaml"
DEFAULT_EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "paths.example.yaml"
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"
PIPELINE_STAGES = ("buildings", "environment", "scene")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.openai_vlm import (  # noqa: E402
    OpenAICompatibleVLM,
    ScenePlan,
    load_scene_plan,
    write_scene_plan,
)


def parse_args() -> argparse.Namespace:
    pipeline_defaults = nested(load_yaml(DEFAULT_PIPELINE_CONFIG), "defaults")
    styles = sorted(path.stem for path in (PROJECT_ROOT / "image_prompts").glob("*.json"))
    parser = argparse.ArgumentParser(
        description=(
            "Run building generation, reusable environment-asset generation, "
            "and Blender city assembly from one layout/depth pair."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Local path configuration. Defaults to configs/paths.local.yaml "
            "when present, otherwise configs/paths.example.yaml."
        ),
    )
    parser.add_argument(
        "--style", choices=styles, default=str(pipeline_defaults["style"])
    )
    parser.add_argument("--case-name", default=str(pipeline_defaults["case_name"]))
    parser.add_argument(
        "--layout-path",
        type=Path,
        default=None,
        help=(
            "Semantic layout input. When --depth-path is omitted, generate a "
            "matching height map and store the canonical pair under output/<case>/inputs."
        ),
    )
    parser.add_argument(
        "--depth-path",
        type=Path,
        default=None,
        help="Existing height/depth input paired with --layout-path.",
    )
    parser.add_argument(
        "--generate-layout",
        action="store_true",
        help="Generate a new layout and height map before running the 3D stages.",
    )
    parser.add_argument("--layout-prompt", default=None)
    parser.add_argument("--height-prompt", default=None)
    parser.add_argument(
        "--scene-prompt",
        default=None,
        help=(
            "High-level city description expanded by an OpenAI-compatible "
            "Scene Design API before generation."
        ),
    )
    parser.add_argument(
        "--scene-plan",
        type=Path,
        default=None,
        help="Reuse a previously generated Scene Plan JSON without an API call.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(pipeline_defaults["stages"]),
        help="Comma-separated subset of: buildings,environment,scene.",
    )
    parser.add_argument("--building-ids", default=str(pipeline_defaults["building_ids"]))
    parser.add_argument(
        "--shape-backend",
        choices=("omni", "2.1"),
        default=str(pipeline_defaults.get("shape_backend", "omni")),
        help="Building shape generator; Omni consumes the sampled point cloud (default).",
    )
    parser.add_argument(
        "--point-samples",
        type=int,
        default=int(pipeline_defaults.get("point_samples", 50)),
        help="Surface point count used for Omni building control (default: 50).",
    )
    parser.add_argument(
        "--layout-palette",
        choices=("citydreamer", "majutsu", "auto"),
        default=str(pipeline_defaults["layout_palette"]),
    )
    parser.add_argument(
        "--edit-profile",
        choices=("legacy", "constrained"),
        default=str(pipeline_defaults["edit_profile"]),
    )
    parser.add_argument(
        "--execution-order",
        choices=("phase", "style"),
        default=str(pipeline_defaults["execution_order"]),
    )
    parser.add_argument(
        "--building-scale-mode",
        choices=("depth", "uniform"),
        default=str(pipeline_defaults["building_scale_mode"]),
        help=(
            "depth matches each building's Z scale to the input height map "
            "(default); uniform preserves the source GLB aspect ratio."
        ),
    )
    parser.add_argument("--seed", type=int, default=int(pipeline_defaults["seed"]))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs instead of resuming from existing files.",
    )
    parser.add_argument(
        "--allow-missing-buildings",
        action="store_true",
        help=(
            "Allow scene assembly to skip missing building GLBs. Intended for "
            "partial smoke tests; full runs remain strict by default."
        ),
    )
    parser.add_argument(
        "--enable-vlm-review",
        action="store_true",
        help="Review generated building, tree and lamp images through the API.",
    )
    parser.add_argument(
        "--vlm-max-attempts",
        type=int,
        default=int(pipeline_defaults.get("vlm_max_attempts", 3)),
        help="Maximum generation attempts for each reviewed image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fully resolved plan without loading models or using GPUs.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required by the pipeline controller. Install requirements/pipeline.txt."
        ) from error

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Copy configs/paths.example.yaml to configs/paths.local.yaml and edit it."
        )
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError("Missing configuration key: " + ".".join(keys))
        value = value[key]
    return value


def optional_nested(
    config: dict[str, Any], *keys: str, default: Any = None
) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def placeholder_scene_plan() -> ScenePlan:
    return ScenePlan(
        **{
            key: f"<Scene Design {key} prompt>"
            for key in (
                "layout",
                "building",
                "tree",
                "lamp",
                "ground",
                "grass",
                "road",
                "water",
                "sky",
            )
        }
    )


def resolve_scene_plan(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[ScenePlan | None, Path | None, str]:
    if args.scene_prompt is not None and args.scene_plan is not None:
        raise ValueError("--scene-prompt and --scene-plan are mutually exclusive.")

    if args.scene_plan is not None:
        plan_path = project_path(args.scene_plan)
        with plan_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        if (
            isinstance(record, dict)
            and isinstance(record.get("style"), str)
            and record["style"] != args.style
        ):
            raise ValueError(
                f"Scene Plan style {record['style']!r} does not match "
                f"--style {args.style!r}: {plan_path}"
            )
        return load_scene_plan(plan_path), plan_path, "provided"

    if args.scene_prompt is None:
        return None, None, "disabled"
    if not args.scene_prompt.strip():
        raise ValueError("--scene-prompt must not be empty.")

    plan_path = PROJECT_ROOT / "output" / args.case_name / "scene_plan.json"
    if plan_path.is_file() and not args.overwrite:
        with plan_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        mismatched = [
            label
            for label, expected in (
                ("source_prompt", args.scene_prompt),
                ("style", args.style),
            )
            if not isinstance(record, dict) or record.get(label) != expected
        ]
        if mismatched:
            raise FileExistsError(
                f"Existing Scene Plan settings differ in {', '.join(mismatched)}: "
                f"{plan_path}\nUse a different --case-name or pass --overwrite."
            )
        print(f"[scene design] Reusing {plan_path}")
        return load_scene_plan(plan_path), plan_path, "cached_api"

    if args.dry_run:
        print(
            "\n[scene design]\n"
            "Would call the configured OpenAI-compatible API and write "
            f"{plan_path}\nNo network request is made during --dry-run."
        )
        return placeholder_scene_plan(), plan_path, "planned_api_dry_run"

    planner = OpenAICompatibleVLM.from_environment(
        prompt_path=project_path(
            optional_nested(
                config,
                "llm",
                "prompt_file",
                default="configs/scene_planner_prompts.json",
            )
        ),
        timeout=float(optional_nested(config, "llm", "timeout", default=120.0)),
        max_retries=int(
            optional_nested(config, "llm", "max_retries", default=2)
        ),
        plan_attempts=int(
            optional_nested(config, "llm", "plan_attempts", default=2)
        ),
    )
    plan = ScenePlan.from_mapping(planner.scene_prompt(args.scene_prompt, args.style))
    write_scene_plan(
        plan_path,
        plan,
        source_prompt=args.scene_prompt,
        style=args.style,
        provider=planner.provider_record(),
    )
    print(f"[scene design] Saved {plan_path}")
    return plan, plan_path, "api"


def project_path(value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return project_path(args.config)
    if DEFAULT_LOCAL_CONFIG.is_file():
        return DEFAULT_LOCAL_CONFIG
    return DEFAULT_EXAMPLE_CONFIG


def parse_stages(spec: str) -> list[str]:
    stages = [item.strip() for item in spec.split(",") if item.strip()]
    if not stages:
        raise ValueError("--stages must select at least one stage.")
    invalid = [stage for stage in stages if stage not in PIPELINE_STAGES]
    if invalid:
        raise ValueError("Unknown pipeline stage(s): " + ", ".join(invalid))
    if len(stages) != len(set(stages)):
        raise ValueError("--stages must not contain duplicates.")
    expected_order = [stage for stage in PIPELINE_STAGES if stage in stages]
    if stages != expected_order:
        raise ValueError("Stages must follow this order: " + ", ".join(PIPELINE_STAGES))
    return stages


def runtime_environment(config: dict[str, Any]) -> dict[str, str]:
    mappings = {
        "MAJUTSUCITY_CONTROLLER_PYTHON": ("paths", "interpreters", "controller"),
        "MAJUTSUCITY_RENDER_PYTHON": ("paths", "interpreters", "render"),
        "MAJUTSUCITY_QWEN_PYTHON": ("paths", "interpreters", "qwen"),
        "MAJUTSUCITY_HUNYUAN_PYTHON": ("paths", "interpreters", "hunyuan"),
        "MAJUTSUCITY_BLENDER_PYTHON": ("paths", "interpreters", "blender"),
        "MAJUTSUCITY_QWEN_IMAGE_EDIT_MODEL": (
            "paths",
            "models",
            "qwen_image_edit",
        ),
        "MAJUTSUCITY_QWEN_IMAGE_MODEL": ("paths", "models", "qwen_image"),
        "MAJUTSUCITY_TEXTURE_LORA": ("paths", "models", "texture_lora"),
        "MAJUTSUCITY_SKYBOX_LORA": ("paths", "models", "skybox_lora"),
        "MAJUTSUCITY_RMBG_MODEL": ("paths", "models", "rmbg"),
        "MAJUTSUCITY_HUNYUAN_MODEL": ("paths", "models", "hunyuan3d"),
        "MAJUTSUCITY_DINO_MODEL": ("paths", "models", "dinov2"),
        "MAJUTSUCITY_HUNYUAN_REPO": ("paths", "external", "hunyuan3d_repo"),
    }
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("MPLCONFIGDIR", "/tmp/majutsucity_mpl")
    environment.setdefault(
        "MAJUTSUCITY_LLM_TIMEOUT",
        str(optional_nested(config, "llm", "timeout", default=120)),
    )
    environment.setdefault(
        "MAJUTSUCITY_LLM_MAX_RETRIES",
        str(optional_nested(config, "llm", "max_retries", default=2)),
    )
    environment.setdefault(
        "MAJUTSUCITY_LLM_PLAN_ATTEMPTS",
        str(optional_nested(config, "llm", "plan_attempts", default=2)),
    )
    for variable, keys in mappings.items():
        environment[variable] = str(project_path(nested(config, *keys)))
    environment["MAJUTSUCITY_HUNYUAN_OMNI_MODEL"] = str(
        project_path(
            optional_nested(
                config,
                "paths",
                "models",
                "hunyuan3d_omni",
                default="models/Hunyuan3D-Omni",
            )
        )
    )
    environment["MAJUTSUCITY_OMNI_DINO_MODEL"] = str(
        project_path(
            optional_nested(
                config,
                "paths",
                "models",
                "dinov2_omni",
                default="models/dinov2-large",
            )
        )
    )
    environment["MAJUTSUCITY_HUNYUAN_OMNI_REPO"] = str(
        project_path(
            optional_nested(
                config,
                "paths",
                "external",
                "hunyuan3d_omni_repo",
                default="external/Hunyuan3D-Omni",
            )
        )
    )
    return environment


def interpreter(config: dict[str, Any], name: str) -> Path:
    return project_path(nested(config, "paths", "interpreters", name))


def print_or_run(
    label: str,
    command: list[str],
    environment: dict[str, str],
    dry_run: bool,
) -> None:
    print(f"\n[{label}]\n{shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def generated_sample(directory: Path, mode: str) -> Path:
    preferred = sorted(directory.glob(f"demo_{mode}_samples_cfg_scale_*.png"))
    candidates = preferred or sorted(directory.glob(f"demo_{mode}_*.png"))
    if not candidates:
        raise FileNotFoundError(
            f"Layout generator did not produce a demo_{mode}_*.png file in {directory}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def quantize_layout_command(
    config: dict[str, Any], source: Path, destination: Path
) -> list[str]:
    return [
        str(interpreter(config, "layout")),
        str(SCRIPT_DIR / "quantize_layout_output.py"),
        "--input-file",
        str(source),
        "--output-file",
        str(destination),
    ]


def height_generation_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    layout_path: Path,
    height_dir: Path,
    height_prompt: str,
) -> list[str]:
    return [
        str(interpreter(config, "layout")),
        str(SCRIPT_DIR / "run_layout_inference.py"),
        "--image_width",
        "512",
        "--image_height",
        "512",
        "--seed",
        str(args.seed),
        "--ddim_steps",
        "50",
        "--unconditional_guidance_scale",
        "9.0",
        "--config_path",
        str(project_path(nested(config, "layout_generation", "height_model_config"))),
        "--model_path",
        str(project_path(nested(config, "layout_generation", "height_checkpoint"))),
        "--result_dir",
        str(height_dir),
        "--use_layout_condition",
        "True",
        "--control_image_path",
        str(layout_path),
        "--text_prompt",
        str(height_prompt),
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_depth_for_layout(
    args: argparse.Namespace,
    config: dict[str, Any],
    environment: dict[str, str],
) -> tuple[Path, Path]:
    source_layout = project_path(args.layout_path)
    input_dir = PROJECT_ROOT / "output" / args.case_name / "inputs"
    layout_path = input_dir / "layout.png"
    depth_path = input_dir / "depth.png"
    source_record_path = input_dir / "layout_source.json"
    height_dir = input_dir / "height_generation"
    height_prompt = args.height_prompt or nested(
        config, "layout_generation", "height_prompt"
    )

    if not args.dry_run and not source_layout.is_file():
        raise FileNotFoundError(f"Layout input not found: {source_layout}")

    source_digest = None if args.dry_run else file_sha256(source_layout)
    if layout_path.is_file() and depth_path.is_file() and not args.overwrite:
        if not source_record_path.is_file():
            raise FileExistsError(
                f"Existing generated input pair has no source record: {input_dir}\n"
                "Use a different --case-name or pass --overwrite after checking the output."
            )
        source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
        expected_record = {
            "sha256": source_digest,
            "height_prompt": str(height_prompt),
            "seed": args.seed,
        }
        mismatched = [
            key
            for key, expected in expected_record.items()
            if source_record.get(key) != expected
        ]
        if mismatched:
            raise FileExistsError(
                f"The supplied layout settings differ from the existing pair in "
                f"{input_dir}: {', '.join(mismatched)}.\n"
                "Use a different --case-name or pass --overwrite after checking the output."
            )
        print(f"[height] Reusing generated depth map in {input_dir}")
        return layout_path, depth_path

    if source_layout != layout_path.resolve():
        print_or_run(
            "layout palette quantization",
            quantize_layout_command(config, source_layout, layout_path),
            environment,
            args.dry_run,
        )
    elif args.dry_run:
        print(f"\n[layout palette quantization]\nReusing canonical layout: {layout_path}")

    height_command = height_generation_command(
        args, config, layout_path, height_dir, str(height_prompt)
    )
    print_or_run("height-map generation", height_command, environment, args.dry_run)
    if args.dry_run:
        return layout_path, depth_path

    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_sample(height_dir, "heightmap"), depth_path)
    source_record = {
        "source_path": str(source_layout),
        "sha256": source_digest,
        "canonical_layout_path": str(layout_path),
        "depth_path": str(depth_path),
        "height_prompt": str(height_prompt),
        "seed": args.seed,
    }
    source_record_path.write_text(
        json.dumps(source_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return layout_path, depth_path


def generate_layout_pair(
    args: argparse.Namespace,
    config: dict[str, Any],
    environment: dict[str, str],
) -> tuple[Path, Path]:
    input_dir = PROJECT_ROOT / "output" / args.case_name / "inputs"
    layout_path = input_dir / "layout.png"
    depth_path = input_dir / "depth.png"
    if layout_path.is_file() and depth_path.is_file() and not args.overwrite:
        print(f"[layout] Reusing existing pair in {input_dir}")
        return layout_path, depth_path

    layout_dir = input_dir / "layout_generation"
    height_dir = input_dir / "height_generation"
    layout_prompt = args.layout_prompt or nested(
        config, "layout_generation", "layout_prompt"
    )
    height_prompt = args.height_prompt or nested(
        config, "layout_generation", "height_prompt"
    )
    common = [
        str(interpreter(config, "layout")),
        str(SCRIPT_DIR / "run_layout_inference.py"),
        "--image_width",
        "512",
        "--image_height",
        "512",
        "--seed",
        str(args.seed),
        "--ddim_steps",
        "50",
        "--unconditional_guidance_scale",
        "9.0",
    ]
    layout_command = [
        *common,
        "--config_path",
        str(
            project_path(
                nested(config, "layout_generation", "layout_model_config")
            )
        ),
        "--longclip_path",
        str(project_path(nested(config, "paths", "models", "longclip"))),
        "--model_path",
        str(project_path(nested(config, "layout_generation", "layout_checkpoint"))),
        "--result_dir",
        str(layout_dir),
        "--use_layout_condition",
        "False",
        "--text_prompt",
        str(layout_prompt),
    ]
    print_or_run("layout generation", layout_command, environment, args.dry_run)
    if args.dry_run:
        expected_layout_sample = (
            layout_dir / "demo_layout_samples_cfg_scale_9.00.png"
        )
        print_or_run(
            "layout palette quantization",
            quantize_layout_command(
                config, expected_layout_sample, layout_path
            ),
            environment,
            True,
        )
        height_command = [
            *common,
            "--config_path",
            str(
                project_path(
                    nested(config, "layout_generation", "height_model_config")
                )
            ),
            "--model_path",
            str(project_path(nested(config, "layout_generation", "height_checkpoint"))),
            "--result_dir",
            str(height_dir),
            "--use_layout_condition",
            "True",
            "--control_image_path",
            str(layout_path),
            "--text_prompt",
            str(height_prompt),
        ]
        print_or_run("height-map generation", height_command, environment, True)
        return layout_path, depth_path

    input_dir.mkdir(parents=True, exist_ok=True)
    raw_layout_path = generated_sample(layout_dir, "layout")
    print_or_run(
        "layout palette quantization",
        quantize_layout_command(config, raw_layout_path, layout_path),
        environment,
        False,
    )
    height_command = [
        *common,
        "--config_path",
        str(
            project_path(
                nested(config, "layout_generation", "height_model_config")
            )
        ),
        "--model_path",
        str(project_path(nested(config, "layout_generation", "height_checkpoint"))),
        "--result_dir",
        str(height_dir),
        "--use_layout_condition",
        "True",
        "--control_image_path",
        str(layout_path),
        "--text_prompt",
        str(height_prompt),
    ]
    print_or_run("height-map generation", height_command, environment, False)
    shutil.copy2(generated_sample(height_dir, "heightmap"), depth_path)
    return layout_path, depth_path


def input_pair(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.depth_path is not None and args.layout_path is None:
        raise ValueError("--depth-path requires --layout-path.")
    if args.layout_path is not None and args.depth_path is None:
        raise ValueError(
            "A layout without a depth map must be resolved through "
            "generate_depth_for_layout()."
        )
    if args.layout_path is not None:
        return project_path(args.layout_path), project_path(args.depth_path)
    return (
        PROJECT_ROOT / "examples" / "pair_cases" / "layout" / f"{args.case_name}.png",
        PROJECT_ROOT / "examples" / "pair_cases" / "depth" / f"{args.case_name}.png",
    )


def building_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    layout_path: Path,
    depth_path: Path,
) -> list[str]:
    command = [
        str(interpreter(config, "controller")),
        str(SCRIPT_DIR / "run_city_asset_pipeline.py"),
        "--style",
        args.style,
        "--case-name",
        args.case_name,
        "--layout-path",
        str(layout_path),
        "--depth-path",
        str(depth_path),
        "--layout-palette",
        args.layout_palette,
        "--building-ids",
        args.building_ids,
        "--shape-backend",
        args.shape_backend,
        "--point-samples",
        str(args.point_samples),
        "--execution-order",
        args.execution_order,
        "--edit-profile",
        args.edit_profile,
        "--qwen-gpu-groups",
        str(nested(config, "runtime", "qwen_gpu_groups")),
        "--hunyuan-gpus",
        str(nested(config, "runtime", "hunyuan_gpus")),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite-images")
    else:
        command.append("--skip-existing-3d")
    if args.resolved_scene_plan is not None:
        command.extend(["--case-prompt", args.resolved_scene_plan.building])
    if args.enable_vlm_review:
        command.extend(
            ["--enable-vlm-review", "--vlm-max-attempts", str(args.vlm_max_attempts)]
        )
    return command


def environment_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    command = [
        str(interpreter(config, "controller")),
        str(SCRIPT_DIR / "run_style_environment_assets.py"),
        "--style",
        args.style,
        "--case-name",
        args.case_name,
        "--qwen-model-path",
        str(project_path(nested(config, "paths", "models", "qwen_image"))),
        "--background-model-path",
        str(project_path(nested(config, "paths", "models", "rmbg"))),
        "--texture-lora-path",
        str(project_path(nested(config, "paths", "models", "texture_lora"))),
        "--skybox-lora-path",
        str(project_path(nested(config, "paths", "models", "skybox_lora"))),
        "--qwen-gpu-groups",
        str(nested(config, "runtime", "qwen_gpu_groups")),
        "--hunyuan-gpus",
        str(nested(config, "runtime", "hunyuan_gpus")),
        "--seed",
        str(args.seed + 7300),
    ]
    if args.scene_plan_path_resolved is not None:
        command.extend(["--scene-plan", str(args.scene_plan_path_resolved)])
    if args.enable_vlm_review:
        command.extend(
            ["--enable-vlm-review", "--vlm-max-attempts", str(args.vlm_max_attempts)]
        )
    command.append("--overwrite" if args.overwrite else "--skip-existing")
    return command


def scene_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    layout_path: Path,
    depth_path: Path,
) -> list[str]:
    command = [
        str(interpreter(config, "controller")),
        str(SCRIPT_DIR / "assemble_city_scene.py"),
        "--style",
        args.style,
        "--case-name",
        args.case_name,
        "--layout-path",
        str(layout_path),
        "--depth-path",
        str(depth_path),
        "--layout-palette",
        args.layout_palette,
        "--building-scale-mode",
        args.building_scale_mode,
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.allow_missing_buildings:
        command.append("--allow-missing-buildings")
    return command


def write_run_manifest(
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    stages: list[str],
    layout_path: Path,
    depth_path: Path,
    input_mode: str,
) -> None:
    manifest_path = PROJECT_ROOT / "output" / args.case_name / "pipeline_run.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    layout_prompt = None
    height_prompt = None
    if input_mode == "generated_layout":
        layout_prompt = args.layout_prompt or nested(
            config, "layout_generation", "layout_prompt"
        )
    if input_mode in ("generated_layout", "provided_layout_generated_depth"):
        height_prompt = args.height_prompt or nested(
            config, "layout_generation", "height_prompt"
        )
    manifest = {
        "case_name": args.case_name,
        "style": args.style,
        "config": str(config_path),
        "layout_path": str(layout_path),
        "depth_path": str(depth_path),
        "input_mode": input_mode,
        "layout_prompt": layout_prompt,
        "height_prompt": height_prompt,
        "scene_prompt": args.scene_prompt,
        "scene_plan_path": (
            str(args.scene_plan_path_resolved)
            if args.scene_plan_path_resolved is not None
            else None
        ),
        "scene_plan_source": args.scene_plan_source,
        "scene_plan": (
            args.resolved_scene_plan.to_dict()
            if args.resolved_scene_plan is not None
            else None
        ),
        "stages": stages,
        "building_ids": args.building_ids,
        "shape_backend": args.shape_backend,
        "point_samples": args.point_samples,
        "building_scale_mode": args.building_scale_mode,
        "allow_missing_buildings": args.allow_missing_buildings,
        "vlm_review": {
            "enabled": args.enable_vlm_review,
            "max_attempts": args.vlm_max_attempts,
        },
        "seed": args.seed,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.case_name):
        raise ValueError("--case-name may only contain letters, numbers, '.', '_' and '-'.")
    stages = parse_stages(args.stages)
    config_path = resolve_config_path(args)
    config = load_yaml(config_path)
    environment = runtime_environment(config)

    if args.vlm_max_attempts <= 0:
        raise ValueError("--vlm-max-attempts must be positive.")
    if args.point_samples <= 0:
        raise ValueError("--point-samples must be positive.")
    (
        args.resolved_scene_plan,
        args.scene_plan_path_resolved,
        args.scene_plan_source,
    ) = resolve_scene_plan(args, config)
    if args.resolved_scene_plan is not None and args.layout_prompt is None:
        args.layout_prompt = args.resolved_scene_plan.layout

    if args.generate_layout:
        if args.layout_path is not None or args.depth_path is not None:
            raise ValueError("--generate-layout cannot be combined with explicit input paths.")
        layout_path, depth_path = generate_layout_pair(args, config, environment)
        input_mode = "generated_layout"
    elif args.layout_path is not None and args.depth_path is None:
        layout_path, depth_path = generate_depth_for_layout(args, config, environment)
        input_mode = "provided_layout_generated_depth"
    else:
        layout_path, depth_path = input_pair(args)
        input_mode = (
            "provided_pair" if args.layout_path is not None else "included_example_pair"
        )

    print("\nResolved MajutsuCity pipeline:")
    print(f"  config:      {config_path}")
    print(f"  case:        {args.case_name}")
    print(f"  style:       {args.style}")
    print(f"  layout:      {layout_path}")
    print(f"  depth:       {depth_path}")
    print(f"  input mode:  {input_mode}")
    print(f"  scene plan:  {args.scene_plan_source}")
    if args.scene_plan_path_resolved is not None:
        print(f"  plan path:   {args.scene_plan_path_resolved}")
    print(f"  VLM review:  {args.enable_vlm_review}")
    print(f"  shape model: {args.shape_backend}")
    if args.shape_backend == "omni":
        print(f"  point count: {args.point_samples}")
    print(f"  scale mode:  {args.building_scale_mode}")
    print(f"  stages:      {', '.join(stages)}")
    print(f"  output:      {PROJECT_ROOT / 'output' / args.case_name}")

    if not args.dry_run:
        missing = [path for path in (layout_path, depth_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing layout input(s):\n  " + "\n  ".join(map(str, missing))
            )

    commands = {
        "buildings": building_command(args, config, layout_path, depth_path),
        "environment": environment_command(args, config),
        "scene": scene_command(args, config, layout_path, depth_path),
    }
    for stage in stages:
        print_or_run(stage, commands[stage], environment, args.dry_run)

    if args.dry_run:
        print("\nDry run completed; no subprocess was started and no model was loaded.")
    else:
        write_run_manifest(
            args,
            config_path,
            config,
            stages,
            layout_path,
            depth_path,
            input_mode,
        )
        print(f"\nPipeline completed: {PROJECT_ROOT / 'output' / args.case_name}")


if __name__ == "__main__":
    main()
