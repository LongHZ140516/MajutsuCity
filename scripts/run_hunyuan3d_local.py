#!/usr/bin/env python3
"""Run local Hunyuan3D-2.1 shape generation and PBR painting."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

from _runtime import env_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUNYUAN_ROOT = env_path(
    "MAJUTSUCITY_HUNYUAN_REPO", "external/Hunyuan3D-2.1"
)
DEFAULT_MODEL_PATH = env_path(
    "MAJUTSUCITY_HUNYUAN_MODEL", "models/Hunyuan3D-2.1"
)
DEFAULT_DINO_PATH = env_path(
    "MAJUTSUCITY_DINO_MODEL", "models/dinov2-giant"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an untextured mesh and/or a textured GLB with the local "
            "Hunyuan3D-2.1 installation."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("shape", "paint", "all"),
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--building-id",
        type=int,
        default=28,
        help="Single building ID, retained for backward compatibility (default: 28).",
    )
    parser.add_argument(
        "--building-ids",
        default=None,
        help=(
            "Comma-separated IDs, a range such as 0-41, or 'all'. "
            "Overrides --building-id."
        ),
    )
    parser.add_argument(
        "--case-name",
        default="SE-Stockholm_64",
        help="Case directory under the repository's output directory.",
    )
    parser.add_argument(
        "--assets-name",
        default="assets",
        help="Asset directory under output/<case-name> (default: assets).",
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=None,
        help="RGBA input image. Defaults to building_<id>/image_woBG.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the selected building directory.",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="Non-negative physical GPU index exposed to this process (default: 0).",
    )
    parser.add_argument(
        "--hunyuan-root",
        type=Path,
        default=DEFAULT_HUNYUAN_ROOT,
        help="Local Hunyuan3D-2.1 source repository.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local Hunyuan3D-2.1 model directory.",
    )
    parser.add_argument(
        "--dino-path",
        type=Path,
        default=DEFAULT_DINO_PATH,
        help="Local DINOv2-Giant snapshot used by the paint stage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shape generation seed (default: 42).",
    )
    parser.add_argument(
        "--shape-steps",
        type=int,
        default=50,
        help="Shape diffusion steps (default: 50).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Shape classifier-free guidance scale (default: 5.0).",
    )
    parser.add_argument(
        "--octree-resolution",
        type=int,
        default=384,
        help="Shape surface extraction resolution (default: 384).",
    )
    parser.add_argument(
        "--paint-views",
        type=int,
        choices=range(6, 10),
        default=6,
        metavar="{6,7,8,9}",
        help="Number of paint views (default: 6).",
    )
    parser.add_argument(
        "--paint-resolution",
        type=int,
        choices=(512, 768),
        default=512,
        help="Paint multiview resolution (default: 512).",
    )
    parser.add_argument(
        "--no-remesh",
        action="store_true",
        help="Disable remeshing before PBR painting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose requested output GLB already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display resolved paths without loading any model.",
    )
    parser.add_argument(
        "--batch-job-file",
        type=Path,
        default=None,
        help=(
            "JSON list of explicit cross-style image/output jobs. The selected "
            "stage model is loaded only once for the entire list."
        ),
    )
    args = parser.parse_args()
    if not args.gpu.isdecimal():
        parser.error("--gpu must be a non-negative physical GPU index.")
    return args


def parse_building_id_spec(args: argparse.Namespace) -> list[int]:
    if not args.building_ids:
        return [args.building_id]

    spec = args.building_ids.strip().lower()
    buildings_root = (
        PROJECT_ROOT
        / "output"
        / args.case_name
        / args.assets_name
        / "buildings"
    )
    if spec == "all":
        building_ids = []
        for path in buildings_root.glob("building_*"):
            try:
                building_id = int(path.name.removeprefix("building_"))
            except ValueError:
                continue
            if (path / "image_woBG.png").is_file():
                building_ids.append(building_id)
        if not building_ids:
            raise FileNotFoundError(
                f"No building_*/image_woBG.png inputs found under {buildings_root}"
            )
        return sorted(building_ids)

    building_ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending building range: {part}")
            building_ids.update(range(start, end + 1))
        else:
            building_ids.add(int(part))
    if not building_ids:
        raise ValueError("No building IDs selected.")
    return sorted(building_ids)


def resolve_paths(args: argparse.Namespace, building_id: int) -> dict[str, Path]:
    building_dir = (
        PROJECT_ROOT
        / "output"
        / args.case_name
        / args.assets_name
        / "buildings"
        / f"building_{building_id}"
    )
    output_dir = (args.output_dir or building_dir).expanduser().resolve()
    input_image = (
        args.input_image or building_dir / "image_woBG.png"
    ).expanduser().resolve()

    return {
        "building_id": building_id,
        "style": None,
        "asset_name": f"building_{building_id}",
        "output_stem": "building",
        "hunyuan_root": args.hunyuan_root.expanduser().resolve(),
        "model_path": args.model_path.expanduser().resolve(),
        "dino_path": args.dino_path.expanduser().resolve(),
        "input_image": input_image,
        "output_dir": output_dir,
        "untextured_glb": output_dir / "building_untextured.glb",
        "textured_obj": output_dir / "building.obj",
        "textured_glb": output_dir / "building.glb",
    }


def load_batch_jobs(args: argparse.Namespace) -> list[dict]:
    batch_job_file = args.batch_job_file.expanduser().resolve()
    with batch_job_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    raw_jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError(f"Expected a non-empty 'jobs' list: {batch_job_file}")

    jobs = []
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ValueError(f"Invalid Hunyuan batch job {index}.")
        building_id = raw_job.get("building_id")
        style = raw_job.get("style")
        asset_name = raw_job.get("asset_name", f"building_{building_id}")
        output_stem = raw_job.get("output_stem", "building")
        input_image_value = raw_job.get("input_image")
        output_dir_value = raw_job.get("output_dir")
        if not isinstance(building_id, int):
            raise ValueError(f"Invalid building_id in Hunyuan batch job {index}.")
        if not isinstance(style, str) or not style:
            raise ValueError(f"Invalid style in Hunyuan batch job {index}.")
        if not isinstance(asset_name, str) or not asset_name:
            raise ValueError(f"Invalid asset_name in Hunyuan batch job {index}.")
        if (
            not isinstance(output_stem, str)
            or not output_stem
            or Path(output_stem).name != output_stem
        ):
            raise ValueError(f"Invalid output_stem in Hunyuan batch job {index}.")
        if not isinstance(input_image_value, str) or not input_image_value:
            raise ValueError(f"Invalid input_image in Hunyuan batch job {index}.")
        if not isinstance(output_dir_value, str) or not output_dir_value:
            raise ValueError(f"Invalid output_dir in Hunyuan batch job {index}.")

        input_image = Path(input_image_value).expanduser().resolve()
        output_dir = Path(output_dir_value).expanduser().resolve()
        jobs.append(
            {
                "building_id": building_id,
                "style": style,
                "asset_name": asset_name,
                "output_stem": output_stem,
                "hunyuan_root": args.hunyuan_root.expanduser().resolve(),
                "model_path": args.model_path.expanduser().resolve(),
                "dino_path": args.dino_path.expanduser().resolve(),
                "input_image": input_image,
                "output_dir": output_dir,
                "untextured_glb": output_dir / f"{output_stem}_untextured.glb",
                "textured_obj": output_dir / f"{output_stem}.obj",
                "textured_glb": output_dir / f"{output_stem}.glb",
            }
        )
    return jobs


def validate_paths(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    required = {
        "Hunyuan source repository": paths["hunyuan_root"],
        "Hunyuan model directory": paths["model_path"],
    }
    # Upstream image generation is only planned, not executed, in a pipeline
    # dry-run. Real shape/paint runs still require the image to exist.
    if not args.dry_run:
        required["input image"] = paths["input_image"]
    if args.stage in ("shape", "all"):
        required["Hunyuan shape package"] = (
            paths["hunyuan_root"] / "hy3dshape" / "hy3dshape"
        )
    if args.stage in ("paint", "all"):
        required["local DINOv2 snapshot"] = paths["dino_path"]
        required["Hunyuan paint package"] = paths["hunyuan_root"] / "hy3dpaint"
        required["RealESRGAN checkpoint"] = (
            paths["hunyuan_root"]
            / "hy3dpaint"
            / "ckpt"
            / "RealESRGAN_x4plus.pth"
        )
        required["Hunyuan paint config"] = (
            paths["hunyuan_root"]
            / "hy3dpaint"
            / "cfgs"
            / "hunyuan-paint-pbr.yaml"
        )
    # In a pipeline dry-run, paint is planned immediately after a planned shape
    # stage, so the intermediate GLB is intentionally absent during validation.
    if args.stage == "paint" and not args.dry_run:
        required["untextured GLB"] = paths["untextured_glb"]

    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))

    model_children = (
        "hunyuan3d-dit-v2-1",
        "hunyuan3d-vae-v2-1",
        "hunyuan3d-paintpbr-v2-1",
    )
    missing_model_parts = [
        name for name in model_children if not (paths["model_path"] / name).is_dir()
    ]
    if missing_model_parts:
        raise FileNotFoundError(
            "Incomplete Hunyuan model directory; missing: "
            + ", ".join(missing_model_parts)
        )


def configure_runtime(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def configure_import_paths(hunyuan_root: Path) -> None:
    # This order matches the upstream demo and ensures hy3dpaint's top-level
    # ``utils`` package wins over unrelated project packages.
    for path in (
        hunyuan_root,
        hunyuan_root / "hy3dshape",
        hunyuan_root / "hy3dpaint",
        hunyuan_root / "hy3dpaint" / "custom_rasterizer",
    ):
        sys.path.insert(0, str(path))


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_shape(args: argparse.Namespace, jobs: list[dict]) -> None:
    import torch
    from PIL import Image
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pending_jobs = [
        paths
        for paths in jobs
        if not (args.skip_existing and paths["untextured_glb"].is_file())
    ]
    if not pending_jobs:
        print("[shape] All selected untextured GLBs already exist; skipping model load.")
        return

    print("[shape] Loading local Hunyuan3D shape model", flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(pending_jobs[0]["model_path"])
    )
    for job_index, paths in enumerate(pending_jobs, start=1):
        building_id = paths["building_id"]
        asset_name = paths.get("asset_name", f"building_{building_id}")
        style_prefix = f"{paths['style']}/" if paths.get("style") else ""
        image = Image.open(paths["input_image"]).convert("RGBA")
        generator = torch.Generator(device="cuda").manual_seed(args.seed + building_id)
        print(
            f"[shape {job_index}/{len(pending_jobs)}] Generating "
            f"{style_prefix}{asset_name}",
            flush=True,
        )
        mesh = pipeline(
            image=image,
            num_inference_steps=args.shape_steps,
            guidance_scale=args.guidance_scale,
            octree_resolution=args.octree_resolution,
            generator=generator,
        )[0]
        mesh.export(paths["untextured_glb"])
        print(f"[shape] Saved: {paths['untextured_glb']}", flush=True)
        del mesh
        del image

    del pipeline
    release_cuda_memory()


def patch_paint_snapshot_download(model_path: Path) -> None:
    """Force the upstream paint loader to use the supplied local model root."""
    import huggingface_hub

    def local_snapshot_download(*_args, **_kwargs) -> str:
        return str(model_path)

    huggingface_hub.snapshot_download = local_snapshot_download


def silence_missing_draco_diagnostic() -> None:
    """Keep Blender's optional missing-Draco check from logging an ERROR."""
    try:
        from io_scene_gltf2.io.com import gltf2_io_draco_compression_extension
    except ImportError:
        return

    original_dll_exists = gltf2_io_draco_compression_extension.dll_exists

    def quiet_dll_exists(_quiet: bool = False) -> bool:
        return original_dll_exists(quiet=True)

    gltf2_io_draco_compression_extension.dll_exists = quiet_dll_exists


def silence_blender_shutdown_memory_report() -> None:
    """Silence bpy allocator diagnostics only after a successful paint export."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stream_fds = {sys.stdout.fileno(), sys.stderr.fileno()}
    except (AttributeError, OSError):
        return

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        for stream_fd in stream_fds:
            os.dup2(devnull_fd, stream_fd)
    finally:
        os.close(devnull_fd)


def run_paint(args: argparse.Namespace, jobs: list[dict]) -> None:
    pending_jobs = [
        paths
        for paths in jobs
        if not (args.skip_existing and paths["textured_glb"].is_file())
    ]
    if not pending_jobs:
        print("[paint] All selected textured GLBs already exist; skipping model load.")
        return

    patch_paint_snapshot_download(pending_jobs[0]["model_path"])

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    silence_missing_draco_diagnostic()

    hunyuan_root = pending_jobs[0]["hunyuan_root"]
    config = Hunyuan3DPaintConfig(
        max_num_view=args.paint_views,
        resolution=args.paint_resolution,
    )
    config.realesrgan_ckpt_path = str(
        hunyuan_root / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    )
    config.multiview_cfg_path = str(
        hunyuan_root / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"
    )
    config.multiview_pretrained_path = str(pending_jobs[0]["model_path"])
    config.dino_ckpt_path = str(pending_jobs[0]["dino_path"])
    config.custom_pipeline = "hunyuanpaintpbr"

    print("[paint] Loading local Hunyuan3D paint model", flush=True)
    pipeline = Hunyuan3DPaintPipeline(config)
    for job_index, paths in enumerate(pending_jobs, start=1):
        building_id = paths["building_id"]
        asset_name = paths.get("asset_name", f"building_{building_id}")
        style_prefix = f"{paths['style']}/" if paths.get("style") else ""
        print(
            f"[paint {job_index}/{len(pending_jobs)}] Generating "
            f"{style_prefix}{asset_name}",
            flush=True,
        )
        pipeline(
            mesh_path=str(paths["untextured_glb"]),
            image_path=str(paths["input_image"]),
            output_mesh_path=str(paths["textured_obj"]),
            use_remesh=not args.no_remesh,
            save_glb=True,
        )

        if not paths["textured_glb"].is_file():
            raise RuntimeError(f"Paint stage did not create {paths['textured_glb']}")
        print(f"[paint] Saved: {paths['textured_glb']}", flush=True)

    del pipeline
    release_cuda_memory()


def print_summary(args: argparse.Namespace, jobs: list[dict]) -> None:
    print("Resolved Hunyuan3D job:")
    print(f"  stage:          {args.stage}")
    print(f"  physical GPU:   {args.gpu}")
    print(f"  model path:     {jobs[0]['model_path']}")
    print(f"  building count: {len(jobs)}")
    print(f"  building IDs:   {[job['building_id'] for job in jobs]}")
    for paths in jobs:
        style_prefix = f"{paths['style']}/" if paths.get("style") else ""
        print(
            f"  {style_prefix}building_{paths['building_id']}: "
            f"{paths['input_image']} -> {paths['textured_glb']}"
        )


def main() -> None:
    args = parse_args()
    if (
        not args.assets_name
        or args.assets_name in (".", "..")
        or Path(args.assets_name).name != args.assets_name
    ):
        raise ValueError(
            "--assets-name must be one directory name, for example assets_mc"
        )
    if args.batch_job_file:
        if args.input_image is not None or args.output_dir is not None:
            raise ValueError(
                "--input-image and --output-dir cannot be combined with "
                "--batch-job-file."
            )
        jobs = load_batch_jobs(args)
    else:
        building_ids = parse_building_id_spec(args)
        if len(building_ids) > 1 and (
            args.input_image is not None or args.output_dir is not None
        ):
            raise ValueError(
                "--input-image and --output-dir can only be used with a "
                "single building."
            )
        jobs = [resolve_paths(args, building_id) for building_id in building_ids]
    for paths in jobs:
        validate_paths(args, paths)
    print_summary(args, jobs)
    if args.dry_run:
        print("Dry run passed; no model was loaded.")
        return

    configure_runtime(args.gpu)
    configure_import_paths(jobs[0]["hunyuan_root"])
    for paths in jobs:
        paths["output_dir"].mkdir(parents=True, exist_ok=True)

    if args.stage in ("shape", "all"):
        run_shape(args, jobs)
    if args.stage in ("paint", "all"):
        run_paint(args, jobs)
        silence_blender_shutdown_memory_report()


if __name__ == "__main__":
    main()
