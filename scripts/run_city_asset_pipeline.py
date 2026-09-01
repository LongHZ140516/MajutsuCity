#!/usr/bin/env python3
"""Style-driven controller for Qwen Image Edit followed by Hunyuan3D."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _runtime import env_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "image_prompts"
QWEN_SCRIPT = PROJECT_ROOT / "scripts" / "run_qwen_image_edit_local.py"
HUNYUAN_SCRIPT = PROJECT_ROOT / "scripts" / "run_hunyuan3d_local.py"
CONTROLLER_PYTHON = env_path(
    "MAJUTSUCITY_CONTROLLER_PYTHON", "envs/majutsucity/bin/python"
)
HUNYUAN_PYTHON = env_path(
    "MAJUTSUCITY_HUNYUAN_PYTHON", "envs/majutsucity/bin/python"
)
DEFAULT_LAYOUT = PROJECT_ROOT / "examples/pair_cases/layout/SE-Stockholm_64.png"
DEFAULT_DEPTH = PROJECT_ROOT / "examples/pair_cases/depth/SE-Stockholm_64.png"


def available_styles() -> dict[str, Path]:
    return {
        path.stem: path.resolve()
        for path in sorted(PROMPT_DIR.glob("*.json"))
        if path.is_file()
    }


def parse_args() -> argparse.Namespace:
    styles = available_styles()
    parser = argparse.ArgumentParser(
        description=(
            "Select an image style, edit all selected layout buildings with "
            "the corresponding prompt JSON, then generate their textured GLBs."
        )
    )
    parser.add_argument(
        "--style",
        choices=(*styles, "all"),
        help="Style name resolved to image_prompts/<style>.json.",
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available styles and exit.",
    )
    parser.add_argument(
        "--building-ids",
        default="all",
        help="Comma-separated IDs, a range such as 0-41, or 'all' (default).",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help=(
            "Output case name. Defaults to the layout stem, for example "
            "SE-Stockholm_64."
        ),
    )
    parser.add_argument(
        "--assets-name",
        default=None,
        help=(
            "Asset directory for one style. Defaults to assets_<style>. "
            "Cannot be used with --style all."
        ),
    )
    parser.add_argument("--layout-path", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--depth-path", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument(
        "--layout-palette",
        choices=("citydreamer", "majutsu", "auto"),
        default="citydreamer",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "qwen", "hunyuan"),
        default="all",
        help="Run the full pipeline or only one major stage (default: all).",
    )
    parser.add_argument(
        "--execution-order",
        choices=("phase", "style"),
        default="phase",
        help=(
            "Use persistent models across all styles (phase, default), or "
            "finish one style at a time (style)."
        ),
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Force Qwen to overwrite existing renders and edited images.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Reuse existing layout-derived rgb.png/depth.png files.",
    )
    parser.add_argument(
        "--skip-existing-3d",
        action="store_true",
        help="Keep existing Hunyuan output GLBs when resuming.",
    )
    parser.add_argument(
        "--qwen-gpu",
        default="0,1",
        help="One two-GPU Qwen group, retained for backward compatibility.",
    )
    parser.add_argument(
        "--qwen-gpu-groups",
        default=None,
        help=(
            "Semicolon-separated two-GPU Qwen workers, for example "
            "'0,1;2,3;4,5'. Overrides --qwen-gpu."
        ),
    )
    parser.add_argument(
        "--hunyuan-gpus",
        default="0,1",
        help=(
            "Comma-separated physical GPUs for parallel Hunyuan workers "
            "(default: 0,1)."
        ),
    )
    parser.add_argument(
        "--hunyuan-gpu",
        default=None,
        help=(
            "Deprecated single-GPU alias. When supplied, it overrides "
            "--hunyuan-gpus."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--point-samples",
        type=int,
        default=50,
        help="Surface point count used by Hunyuan3D-Omni (default: 50).",
    )
    parser.add_argument("--qwen-steps", type=int, default=40)
    parser.add_argument("--qwen-cfg", type=float, default=4.0)
    parser.add_argument(
        "--edit-profile",
        choices=("legacy", "constrained"),
        default="legacy",
        help=(
            "Qwen edit policy: legacy uses the original case-level style prompt "
            "and permits richer redesign; constrained strongly preserves the gray "
            "massing with building_i -> prompt_i and cycles prompts when the layout "
            "contains more buildings (default: legacy)."
        ),
    )
    parser.add_argument(
        "--case-prompt",
        default=None,
        help=(
            "Override the built-in case-level style prompt used by "
            "--edit-profile legacy."
        ),
    )
    parser.add_argument(
        "--enable-vlm-review",
        action="store_true",
        help="Review generated building images through an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--vlm-max-attempts",
        type=int,
        default=3,
        help="Maximum Qwen generation attempts for each reviewed building.",
    )
    parser.add_argument("--shape-steps", type=int, default=50)
    parser.add_argument(
        "--shape-backend",
        choices=("omni", "2.1"),
        default="omni",
        help="Shape generator; Omni uses the sampled building point cloud (default).",
    )
    parser.add_argument("--shape-guidance", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument(
        "--paint-views",
        type=int,
        choices=range(6, 10),
        default=6,
        metavar="{6,7,8,9}",
    )
    parser.add_argument(
        "--paint-resolution",
        type=int,
        choices=(512, 768),
        default=512,
    )
    parser.add_argument("--no-remesh", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate style mapping and print child commands without loading models.",
    )
    args = parser.parse_args()

    if args.list_styles:
        for style, path in styles.items():
            with path.open("r", encoding="utf-8") as file:
                prompt_count = len(json.load(file))
            print(f"{style:12s} {prompt_count:3d} prompts  {path}")
        raise SystemExit(0)
    if not args.style:
        parser.error("--style is required unless --list-styles is used.")
    if args.style == "all" and args.assets_name:
        parser.error("--assets-name cannot be combined with --style all.")
    if args.hunyuan_gpu:
        args.hunyuan_gpus = args.hunyuan_gpu
    if args.vlm_max_attempts <= 0:
        parser.error("--vlm-max-attempts must be positive.")
    if args.point_samples <= 0:
        parser.error("--point-samples must be positive.")
    return args


def validate_prompt_file(prompt_file: Path) -> list[str]:
    with prompt_file.open("r", encoding="utf-8") as file:
        prompts = json.load(file)
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"Expected a JSON list of strings: {prompt_file}")
    return prompts


def parse_hunyuan_gpus(gpu_spec: str) -> list[str]:
    gpu_ids = [item.strip() for item in gpu_spec.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--hunyuan-gpus did not select any GPU.")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--hunyuan-gpus must not contain duplicate GPU IDs.")
    invalid = [gpu_id for gpu_id in gpu_ids if not gpu_id.isdecimal()]
    if invalid:
        raise ValueError(
            "Only non-negative physical GPU IDs are permitted; invalid values: "
            + ", ".join(invalid)
        )
    return gpu_ids


def parse_qwen_gpu_groups(args: argparse.Namespace) -> list[str]:
    group_spec = args.qwen_gpu_groups or args.qwen_gpu
    groups = [group.strip() for group in group_spec.split(";") if group.strip()]
    if not groups:
        raise ValueError("No Qwen GPU group was selected.")

    used_gpu_ids: set[str] = set()
    normalized_groups = []
    for group in groups:
        gpu_ids = [item.strip() for item in group.split(",") if item.strip()]
        if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
            raise ValueError(
                "Every Qwen worker needs exactly two distinct GPUs; "
                f"invalid group: {group!r}"
            )
        invalid = [gpu_id for gpu_id in gpu_ids if not gpu_id.isdecimal()]
        if invalid:
            raise ValueError(
                "Qwen GPU groups only accept non-negative physical IDs; invalid values: "
                + ", ".join(invalid)
            )
        overlap = used_gpu_ids.intersection(gpu_ids)
        if overlap:
            raise ValueError(
                "Qwen GPU groups must be disjoint; repeated IDs: "
                + ", ".join(sorted(overlap))
            )
        used_gpu_ids.update(gpu_ids)
        normalized_groups.append(",".join(gpu_ids))
    return normalized_groups


def resolve_building_ids(args: argparse.Namespace) -> list[int]:
    # Reuse exactly the same layout counting and ID parsing policy as Qwen.
    from run_qwen_image_edit_local import layout_building_ids, parse_id_spec

    valid_ids = layout_building_ids(
        args.layout_path, palette_name=args.layout_palette
    )
    return parse_id_spec(args.building_ids, len(valid_ids), valid_ids=valid_ids)


def shard_building_ids(
    building_ids: list[int],
    gpu_ids: list[str],
) -> list[tuple[str, list[int]]]:
    # Round-robin gives each persistent model worker nearly the same job count
    # while keeping every building assigned to exactly one physical GPU.
    shards = [
        (gpu_id, building_ids[index :: len(gpu_ids)])
        for index, gpu_id in enumerate(gpu_ids)
    ]
    return [(gpu_id, shard) for gpu_id, shard in shards if shard]


def qwen_command(
    args: argparse.Namespace,
    prompt_file: Path,
    case_name: str,
    assets_name: str,
    gpu_group: str,
    building_ids: list[int],
) -> list[str]:
    command = [
        str(CONTROLLER_PYTHON),
        str(QWEN_SCRIPT),
        "--building-ids",
        ",".join(str(building_id) for building_id in building_ids),
        "--case-name",
        case_name,
        "--assets-name",
        assets_name,
        "--layout-path",
        str(args.layout_path),
        "--depth-path",
        str(args.depth_path),
        "--layout-palette",
        args.layout_palette,
        "--prompt-file",
        str(prompt_file),
        "--gpu",
        gpu_group,
        "--seed",
        str(args.seed),
        "--shape-backend",
        args.shape_backend,
        "--point-samples",
        str(args.point_samples),
        "--steps",
        str(args.qwen_steps),
        "--true-cfg-scale",
        str(args.qwen_cfg),
        "--edit-profile",
        args.edit_profile,
    ]
    if args.case_prompt:
        command.extend(["--case-prompt", args.case_prompt])
    if args.enable_vlm_review:
        command.extend(
            ["--enable-vlm-review", "--vlm-max-attempts", str(args.vlm_max_attempts)]
        )
    if args.overwrite_images:
        command.append("--overwrite")
    if args.skip_render:
        command.append("--skip-render")
    return command


def qwen_batch_command(
    args: argparse.Namespace,
    batch_job_file: Path,
    gpu_group: str,
) -> list[str]:
    command = [
        str(CONTROLLER_PYTHON),
        str(QWEN_SCRIPT),
        "--batch-job-file",
        str(batch_job_file),
        "--layout-path",
        str(args.layout_path),
        "--depth-path",
        str(args.depth_path),
        "--layout-palette",
        args.layout_palette,
        "--gpu",
        gpu_group,
        "--seed",
        str(args.seed),
        "--shape-backend",
        args.shape_backend,
        "--point-samples",
        str(args.point_samples),
        "--steps",
        str(args.qwen_steps),
        "--true-cfg-scale",
        str(args.qwen_cfg),
        "--edit-profile",
        args.edit_profile,
    ]
    if args.case_prompt:
        command.extend(["--case-prompt", args.case_prompt])
    if args.enable_vlm_review:
        command.extend(
            ["--enable-vlm-review", "--vlm-max-attempts", str(args.vlm_max_attempts)]
        )
    if args.overwrite_images:
        command.append("--overwrite")
    if args.skip_render:
        command.append("--skip-render")
    return command


def hunyuan_command(
    args: argparse.Namespace,
    case_name: str,
    assets_name: str,
    gpu_id: str,
    building_ids: list[int],
) -> list[str]:
    command = [
        str(HUNYUAN_PYTHON),
        str(HUNYUAN_SCRIPT),
        "--building-ids",
        ",".join(str(building_id) for building_id in building_ids),
        "--case-name",
        case_name,
        "--assets-name",
        assets_name,
        "--gpu",
        gpu_id,
        "--stage",
        "all",
        "--shape-backend",
        args.shape_backend,
        "--seed",
        str(args.seed),
        "--shape-steps",
        str(args.shape_steps),
        "--guidance-scale",
        str(args.shape_guidance),
        "--octree-resolution",
        str(args.octree_resolution),
        "--paint-views",
        str(args.paint_views),
        "--paint-resolution",
        str(args.paint_resolution),
    ]
    if args.no_remesh:
        command.append("--no-remesh")
    if args.skip_existing_3d:
        command.append("--skip-existing")
    return command


def hunyuan_batch_command(
    args: argparse.Namespace,
    batch_job_file: Path,
    gpu_id: str,
    stage: str,
) -> list[str]:
    command = [
        str(HUNYUAN_PYTHON),
        str(HUNYUAN_SCRIPT),
        "--batch-job-file",
        str(batch_job_file),
        "--gpu",
        gpu_id,
        "--stage",
        stage,
        "--shape-backend",
        args.shape_backend,
        "--seed",
        str(args.seed),
        "--shape-steps",
        str(args.shape_steps),
        "--guidance-scale",
        str(args.shape_guidance),
        "--octree-resolution",
        str(args.octree_resolution),
        "--paint-views",
        str(args.paint_views),
        "--paint-resolution",
        str(args.paint_resolution),
    ]
    if args.no_remesh:
        command.append("--no-remesh")
    if args.skip_existing_3d:
        command.append("--skip-existing")
    return command


def write_manifest(
    case_name: str,
    assets_name: str,
    style: str,
    args: argparse.Namespace,
    prompt_file: Path,
    prompts: list[str],
    status: str,
) -> None:
    assets_dir = PROJECT_ROOT / "output" / case_name / assets_name
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case_name": case_name,
        "assets_name": assets_name,
        "style": style,
        "prompt_file": str(prompt_file),
        "prompt_count": len(prompts),
        "building_ids": args.building_ids,
        "resolved_building_ids": args.resolved_building_ids,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "qwen": {
            "edit_profile": args.edit_profile,
            "case_prompt": args.case_prompt,
            "vlm_review": {
                "enabled": args.enable_vlm_review,
                "max_attempts": args.vlm_max_attempts,
            },
            "gpu_groups": args.qwen_gpu_groups_resolved,
            "shards": {
                gpu_group: building_ids
                for gpu_group, building_ids in shard_building_ids(
                    args.resolved_building_ids,
                    args.qwen_gpu_groups_resolved,
                )
            },
            "seed": args.seed,
            "steps": args.qwen_steps,
            "true_cfg_scale": args.qwen_cfg,
        },
        "hunyuan": {
            "shape_backend": args.shape_backend,
            "point_samples": args.point_samples,
            "gpus": args.hunyuan_gpu_ids,
            "shards": {
                gpu_id: building_ids
                for gpu_id, building_ids in shard_building_ids(
                    args.resolved_building_ids,
                    args.hunyuan_gpu_ids,
                )
            },
            "shape_steps": args.shape_steps,
            "guidance_scale": args.shape_guidance,
            "octree_resolution": args.octree_resolution,
            "paint_views": args.paint_views,
            "paint_resolution": args.paint_resolution,
        },
    }
    with (assets_dir / "pipeline_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def run_checked(command: list[str], stage_name: str) -> None:
    print(f"\n[{stage_name}] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_parallel_checked(
    commands: list[tuple[str, list[str]]],
    stage_name: str,
) -> None:
    processes: list[tuple[str, list[str], subprocess.Popen]] = []
    try:
        for worker_name, command in commands:
            print(
                f"\n[{stage_name} / {worker_name}] {shlex.join(command)}",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                start_new_session=True,
            )
            processes.append((worker_name, command, process))

        pending = set(range(len(processes)))
        while pending:
            for index in list(pending):
                worker_name, command, process = processes[index]
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(index)
                if return_code != 0:
                    for other_index in pending:
                        other_process = processes[other_index][2]
                        if other_process.poll() is None:
                            os.killpg(other_process.pid, signal.SIGTERM)
                    for other_index in pending:
                        processes[other_index][2].wait()
                    raise subprocess.CalledProcessError(return_code, command)
                print(f"[{stage_name} / {worker_name}] completed", flush=True)
            if pending:
                time.sleep(0.5)
    except BaseException:
        for _worker_name, _command, process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _worker_name, _command, process in processes:
            if process.poll() is None:
                process.wait()
        raise


def run_style(
    args: argparse.Namespace,
    style: str,
    prompt_file: Path,
    case_name: str,
    assets_name: str,
) -> None:
    prompts = validate_prompt_file(prompt_file)
    qwen_shards = shard_building_ids(
        args.resolved_building_ids,
        args.qwen_gpu_groups_resolved,
    )
    qwen_commands = [
        (
            f"GPUs {gpu_group}: {','.join(str(item) for item in building_ids)}",
            qwen_command(
                args,
                prompt_file,
                case_name,
                assets_name,
                gpu_group,
                building_ids,
            ),
        )
        for gpu_group, building_ids in qwen_shards
    ]
    hunyuan_shards = shard_building_ids(
        args.resolved_building_ids,
        args.hunyuan_gpu_ids,
    )
    hunyuan_commands = [
        (
            f"GPU {gpu_id}: {','.join(str(item) for item in building_ids)}",
            hunyuan_command(
                args,
                case_name,
                assets_name,
                gpu_id,
                building_ids,
            ),
        )
        for gpu_id, building_ids in hunyuan_shards
    ]
    assets_dir = PROJECT_ROOT / "output" / case_name / assets_name

    print("\nResolved city asset pipeline:")
    print(f"  style:         {style}")
    print(f"  prompt file:   {prompt_file}")
    print(f"  prompt count:  {len(prompts)}")
    print(f"  building IDs:  {args.building_ids}")
    print(f"  assets output: {assets_dir}")
    print(f"  major stage:   {args.stage}")
    print(f"  Qwen groups:   {'; '.join(args.qwen_gpu_groups_resolved)}")
    for worker_name, command in qwen_commands:
        print(f"  Qwen {worker_name}: {shlex.join(command)}")
    print(f"  Hunyuan GPUs:  {', '.join(args.hunyuan_gpu_ids)}")
    for worker_name, command in hunyuan_commands:
        print(f"  Hunyuan {worker_name}: {shlex.join(command)}")

    if args.dry_run:
        if args.stage in ("all", "qwen"):
            run_parallel_checked(
                [
                    (worker_name, [*command, "--dry-run"])
                    for worker_name, command in qwen_commands
                ],
                f"Qwen dry-run: {style}",
            )
        if args.stage == "hunyuan":
            run_parallel_checked(
                [
                    (worker_name, [*command, "--dry-run"])
                    for worker_name, command in hunyuan_commands
                ],
                f"Hunyuan3D dry-run: {style}",
            )
        return

    write_manifest(
        case_name, assets_name, style, args, prompt_file, prompts, status="started"
    )
    try:
        if args.stage in ("all", "qwen"):
            run_parallel_checked(qwen_commands, f"Qwen Image Edit: {style}")
            write_manifest(
                case_name,
                assets_name,
                style,
                args,
                prompt_file,
                prompts,
                status="qwen_completed",
            )

        # subprocess.run(check=True) guarantees Hunyuan starts only after all
        # selected Qwen edits and background removals finish successfully. The
        # building shards then run concurrently on their assigned physical GPUs.
        if args.stage in ("all", "hunyuan"):
            run_parallel_checked(hunyuan_commands, f"Hunyuan3D: {style}")
            write_manifest(
                case_name,
                assets_name,
                style,
                args,
                prompt_file,
                prompts,
                status="completed",
            )
    except Exception:
        write_manifest(
            case_name, assets_name, style, args, prompt_file, prompts, status="failed"
        )
        raise

    print(f"Style completed: {style} -> {assets_dir}", flush=True)


def write_job_file(path: Path, jobs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"version": 1, "jobs": jobs}, file, ensure_ascii=False, indent=2)


def update_style_manifests(
    args: argparse.Namespace,
    selected_styles: list[str],
    styles: dict[str, Path],
    case_name: str,
    status: str,
) -> None:
    for style in selected_styles:
        assets_name = args.assets_name or f"assets_{style}"
        prompts = validate_prompt_file(styles[style])
        write_manifest(
            case_name,
            assets_name,
            style,
            args,
            styles[style],
            prompts,
            status=status,
        )


def build_phase_commands(
    args: argparse.Namespace,
    selected_styles: list[str],
    styles: dict[str, Path],
    case_name: str,
) -> tuple[
    list[tuple[str, list[str]]],
    list[tuple[str, list[str]]],
    list[tuple[str, list[str]]],
]:
    job_root = PROJECT_ROOT / "output" / case_name / ".pipeline_jobs"

    qwen_commands = []
    for worker_index, (gpu_group, building_ids) in enumerate(
        shard_building_ids(
            args.resolved_building_ids,
            args.qwen_gpu_groups_resolved,
        )
    ):
        jobs = [
            {
                "style": style,
                "case_name": case_name,
                "assets_name": args.assets_name or f"assets_{style}",
                "prompt_file": str(styles[style]),
                "building_ids": building_ids,
            }
            for style in selected_styles
        ]
        job_file = job_root / f"qwen_worker_{worker_index}.json"
        write_job_file(job_file, jobs)
        qwen_commands.append(
            (
                f"GPUs {gpu_group}",
                qwen_batch_command(args, job_file, gpu_group),
            )
        )

    flattened_hunyuan_jobs = []
    for style in selected_styles:
        assets_name = args.assets_name or f"assets_{style}"
        for building_id in args.resolved_building_ids:
            output_dir = (
                PROJECT_ROOT
                / "output"
                / case_name
                / assets_name
                / "buildings"
                / f"building_{building_id}"
            ).resolve()
            flattened_hunyuan_jobs.append(
                {
                    "style": style,
                    "building_id": building_id,
                    "input_image": str(output_dir / "image_woBG.png"),
                    "point_cloud": str(output_dir / "points_ascii.ply"),
                    "output_dir": str(output_dir),
                }
            )

    hunyuan_shape_commands = []
    hunyuan_paint_commands = []
    for worker_index, gpu_id in enumerate(args.hunyuan_gpu_ids):
        jobs = flattened_hunyuan_jobs[worker_index :: len(args.hunyuan_gpu_ids)]
        if not jobs:
            continue
        job_file = job_root / f"hunyuan_gpu_{gpu_id}.json"
        write_job_file(job_file, jobs)
        worker_name = f"GPU {gpu_id} ({len(jobs)} jobs)"
        hunyuan_shape_commands.append(
            (
                worker_name,
                hunyuan_batch_command(args, job_file, gpu_id, "shape"),
            )
        )
        hunyuan_paint_commands.append(
            (
                worker_name,
                hunyuan_batch_command(args, job_file, gpu_id, "paint"),
            )
        )

    return qwen_commands, hunyuan_shape_commands, hunyuan_paint_commands


def run_phase_order(
    args: argparse.Namespace,
    selected_styles: list[str],
    styles: dict[str, Path],
    case_name: str,
) -> None:
    qwen_commands, shape_commands, paint_commands = build_phase_commands(
        args,
        selected_styles,
        styles,
        case_name,
    )

    total_jobs = len(selected_styles) * len(args.resolved_building_ids)
    print("\nResolved phase-ordered pipeline:")
    print(f"  styles:          {', '.join(selected_styles)}")
    print(f"  total assets:    {total_jobs}")
    print(f"  Qwen model loads:{len(qwen_commands)} total (one per GPU pair)")
    print(f"  Shape loads:     {len(shape_commands)} total (one per GPU)")
    print(f"  Paint loads:     {len(paint_commands)} total (one per GPU)")
    for worker_name, command in qwen_commands:
        print(f"  Qwen {worker_name}: {shlex.join(command)}")
    for worker_name, command in shape_commands:
        print(f"  Shape {worker_name}: {shlex.join(command)}")
    for worker_name, command in paint_commands:
        print(f"  Paint {worker_name}: {shlex.join(command)}")

    if args.dry_run:
        if args.stage in ("all", "qwen"):
            run_parallel_checked(
                [
                    (worker_name, [*command, "--dry-run"])
                    for worker_name, command in qwen_commands
                ],
                "Persistent Qwen dry-run",
            )
        if args.stage == "hunyuan":
            run_parallel_checked(
                [
                    (worker_name, [*command, "--dry-run"])
                    for worker_name, command in shape_commands
                ],
                "Cross-style Shape dry-run",
            )
        print("Phase-ordered dry run passed; no generation model was loaded.")
        return

    update_style_manifests(
        args,
        selected_styles,
        styles,
        case_name,
        status="started",
    )
    try:
        if args.stage in ("all", "qwen"):
            run_parallel_checked(qwen_commands, "Persistent Qwen Image Edit")
            update_style_manifests(
                args,
                selected_styles,
                styles,
                case_name,
                status="qwen_completed",
            )

        if args.stage in ("all", "hunyuan"):
            run_parallel_checked(shape_commands, "Cross-style Hunyuan Shape")
            update_style_manifests(
                args,
                selected_styles,
                styles,
                case_name,
                status="shape_completed",
            )
            run_parallel_checked(paint_commands, "Cross-style Hunyuan Paint")
            update_style_manifests(
                args,
                selected_styles,
                styles,
                case_name,
                status="completed",
            )
    except Exception:
        update_style_manifests(
            args,
            selected_styles,
            styles,
            case_name,
            status="failed",
        )
        raise

    print(
        f"Phase-ordered pipeline completed: {PROJECT_ROOT / 'output' / case_name}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.layout_path = args.layout_path.expanduser().resolve()
    args.depth_path = args.depth_path.expanduser().resolve()
    args.qwen_gpu_groups_resolved = parse_qwen_gpu_groups(args)
    args.hunyuan_gpu_ids = parse_hunyuan_gpus(args.hunyuan_gpus)
    args.resolved_building_ids = resolve_building_ids(args)

    styles = available_styles()
    selected_styles = list(styles) if args.style == "all" else [args.style]
    case_name = args.case_name or args.layout_path.stem

    required = {
        "layout": args.layout_path,
        "depth": args.depth_path,
        "Qwen script": QWEN_SCRIPT,
        "Hunyuan script": HUNYUAN_SCRIPT,
        "controller Python": CONTROLLER_PYTHON,
        "Hunyuan Python": HUNYUAN_PYTHON,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))

    if args.execution_order == "phase":
        run_phase_order(args, selected_styles, styles, case_name)
    else:
        print(
            f"Selected {len(selected_styles)} style(s), processed sequentially: "
            f"{', '.join(selected_styles)}"
        )
        for style in selected_styles:
            assets_name = args.assets_name or f"assets_{style}"
            run_style(args, style, styles[style], case_name, assets_name)

        if args.dry_run:
            print("Controller dry run passed; no generation model was loaded.")
        else:
            print(
                f"All selected styles completed: "
                f"{PROJECT_ROOT / 'output' / case_name}",
                flush=True,
            )


if __name__ == "__main__":
    main()
