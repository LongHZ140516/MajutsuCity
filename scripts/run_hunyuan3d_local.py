#!/usr/bin/env python3
"""Run Hunyuan3D-Omni point-controlled shape and Hunyuan3D-2.1 painting."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

from _runtime import env_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OMNI_ROOT = env_path(
    "MAJUTSUCITY_HUNYUAN_OMNI_REPO", "external/Hunyuan3D-Omni"
)
DEFAULT_OMNI_MODEL_PATH = env_path(
    "MAJUTSUCITY_HUNYUAN_OMNI_MODEL", "models/Hunyuan3D-Omni"
)
DEFAULT_OMNI_DINO_PATH = env_path(
    "MAJUTSUCITY_OMNI_DINO_MODEL", "models/dinov2-large"
)
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
            "Generate an untextured mesh with Hunyuan3D-Omni point control "
            "(or the Hunyuan3D-2.1 fallback), then paint it with Hunyuan3D-2.1."
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
        "--shape-backend",
        choices=("omni", "2.1"),
        default="omni",
        help="Shape model backend (default: omni).",
    )
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=None,
        help=(
            "Point-cloud control for a single Omni job. Defaults to "
            "building_<id>/points_ascii.ply."
        ),
    )
    parser.add_argument(
        "--omni-root",
        type=Path,
        default=DEFAULT_OMNI_ROOT,
        help="Local Hunyuan3D-Omni source repository.",
    )
    parser.add_argument(
        "--omni-model-path",
        type=Path,
        default=DEFAULT_OMNI_MODEL_PATH,
        help="Local Hunyuan3D-Omni model directory.",
    )
    parser.add_argument(
        "--omni-dino-path",
        type=Path,
        default=DEFAULT_OMNI_DINO_PATH,
        help="Local DINOv2-Large snapshot used by Hunyuan3D-Omni.",
    )
    parser.add_argument(
        "--hunyuan-root",
        type=Path,
        default=DEFAULT_HUNYUAN_ROOT,
        help="Local Hunyuan3D-2.1 source repository used for painting.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local Hunyuan3D-2.1 model directory used for painting.",
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
        "shape_backend": args.shape_backend,
        "omni_root": args.omni_root.expanduser().resolve(),
        "omni_model_path": args.omni_model_path.expanduser().resolve(),
        "omni_dino_path": args.omni_dino_path.expanduser().resolve(),
        "hunyuan_root": args.hunyuan_root.expanduser().resolve(),
        "model_path": args.model_path.expanduser().resolve(),
        "dino_path": args.dino_path.expanduser().resolve(),
        "input_image": input_image,
        "point_cloud": (
            args.point_cloud or building_dir / "points_ascii.ply"
        ).expanduser().resolve(),
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
        point_cloud_value = raw_job.get("point_cloud")
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
        if point_cloud_value is not None and (
            not isinstance(point_cloud_value, str) or not point_cloud_value
        ):
            raise ValueError(f"Invalid point_cloud in Hunyuan batch job {index}.")
        if not isinstance(output_dir_value, str) or not output_dir_value:
            raise ValueError(f"Invalid output_dir in Hunyuan batch job {index}.")

        input_image = Path(input_image_value).expanduser().resolve()
        output_dir = Path(output_dir_value).expanduser().resolve()
        point_cloud = Path(
            point_cloud_value or output_dir / "points_ascii.ply"
        ).expanduser().resolve()
        jobs.append(
            {
                "building_id": building_id,
                "style": style,
                "asset_name": asset_name,
                "output_stem": output_stem,
                "shape_backend": args.shape_backend,
                "omni_root": args.omni_root.expanduser().resolve(),
                "omni_model_path": args.omni_model_path.expanduser().resolve(),
                "omni_dino_path": args.omni_dino_path.expanduser().resolve(),
                "hunyuan_root": args.hunyuan_root.expanduser().resolve(),
                "model_path": args.model_path.expanduser().resolve(),
                "dino_path": args.dino_path.expanduser().resolve(),
                "input_image": input_image,
                "point_cloud": point_cloud,
                "output_dir": output_dir,
                "untextured_glb": output_dir / f"{output_stem}_untextured.glb",
                "textured_obj": output_dir / f"{output_stem}.obj",
                "textured_glb": output_dir / f"{output_stem}.glb",
            }
        )
    return jobs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_path(paths: dict[str, Path], kind: str) -> Path:
    return paths["output_dir"] / f"{paths['output_stem']}_{kind}.json"


def load_record(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return record if isinstance(record, dict) else None


def write_record(path: Path, record: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2, sort_keys=True)


def expected_shape_record(args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    record = {
        "backend": args.shape_backend,
        "input_image": str(paths["input_image"]),
        "input_image_sha256": file_sha256(paths["input_image"]),
        "seed": args.seed + paths["building_id"],
        "shape_steps": args.shape_steps,
        "guidance_scale": args.guidance_scale,
        "octree_resolution": args.octree_resolution,
    }
    if args.shape_backend == "omni":
        record.update(
            {
                "point_cloud": str(paths["point_cloud"]),
                "point_cloud_sha256": file_sha256(paths["point_cloud"]),
                "omni_dino_path": str(paths["omni_dino_path"]),
            }
        )
    return record


def may_reuse_shape(args: argparse.Namespace, paths: dict[str, Path]) -> bool:
    if not args.skip_existing or not paths["untextured_glb"].is_file():
        return False
    # Preserve resumability for assets created by the earlier image-only 2.1
    # pipeline, which predates provenance records.
    if args.shape_backend == "2.1" and not record_path(paths, "shape").is_file():
        return True
    return load_record(record_path(paths, "shape")) == expected_shape_record(args, paths)


def expected_paint_record(args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    return {
        "source_shape_sha256": file_sha256(paths["untextured_glb"]),
        "input_image_sha256": file_sha256(paths["input_image"]),
        "paint_views": args.paint_views,
        "paint_resolution": args.paint_resolution,
        "use_remesh": not args.no_remesh,
    }


def may_reuse_paint(args: argparse.Namespace, paths: dict[str, Path]) -> bool:
    if not args.skip_existing or not paths["textured_glb"].is_file():
        return False
    paint_record = record_path(paths, "paint")
    if not paint_record.is_file():
        # A new shape record means the shape stage is provenance-aware. Never
        # pair it with an unrecorded, potentially stale textured mesh.
        return not record_path(paths, "shape").is_file()
    return load_record(paint_record) == expected_paint_record(args, paths)


def validate_paths(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> None:
    required = {}
    # Upstream image generation is only planned, not executed, in a pipeline
    # dry-run. Real shape/paint runs still require the image to exist.
    if not args.dry_run:
        required["input image"] = paths["input_image"]
    if args.stage in ("shape", "all"):
        if args.shape_backend == "omni":
            required.update(
                {
                    "Hunyuan3D-Omni source repository": paths["omni_root"],
                    "Hunyuan3D-Omni model directory": paths["omni_model_path"],
                    "local DINOv2-Large snapshot": paths["omni_dino_path"],
                    "Hunyuan3D-Omni shape package": (
                        paths["omni_root"]
                        / "hy3dshape"
                        / "pipelines"
                        / "pipeline_generation_sit_omni.py"
                    ),
                }
            )
            if not args.dry_run:
                required["Omni point cloud"] = paths["point_cloud"]
        else:
            required.update(
                {
                    "Hunyuan3D-2.1 source repository": paths["hunyuan_root"],
                    "Hunyuan3D-2.1 model directory": paths["model_path"],
                    "Hunyuan3D-2.1 shape package": (
                        paths["hunyuan_root"] / "hy3dshape" / "hy3dshape"
                    ),
                }
            )
    if args.stage in ("paint", "all"):
        required["Hunyuan3D-2.1 source repository"] = paths["hunyuan_root"]
        required["Hunyuan3D-2.1 model directory"] = paths["model_path"]
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

    if args.stage in ("shape", "all") and args.shape_backend == "omni":
        omni_children = ("model", "vae", "cond_encoder", "scheduler", "image_processor")
        missing_omni_parts = [
            name
            for name in omni_children
            if not (paths["omni_model_path"] / name).is_dir()
        ]
        if missing_omni_parts:
            raise FileNotFoundError(
                "Incomplete Hunyuan3D-Omni model directory; missing: "
                + ", ".join(missing_omni_parts)
            )

    model_children = []
    if args.stage in ("shape", "all") and args.shape_backend == "2.1":
        model_children.extend(("hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1"))
    if args.stage in ("paint", "all"):
        model_children.append("hunyuan3d-paintpbr-v2-1")
    missing_model_parts = [
        name for name in model_children if not (paths["model_path"] / name).is_dir()
    ]
    if missing_model_parts:
        raise FileNotFoundError(
            "Incomplete Hunyuan3D-2.1 model directory; missing: "
            + ", ".join(missing_model_parts)
        )


def configure_runtime(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def configure_import_paths(
    shape_backend: str,
    omni_root: Path,
    hunyuan_root: Path,
) -> None:
    """Expose the selected shape package and the separate 2.1 paint package."""
    shape_paths = (omni_root,) if shape_backend == "omni" else (
        hunyuan_root / "hy3dshape",
        hunyuan_root,
    )
    # Match the upstream 2.1 paint runtime: its top-level ``utils`` package and
    # custom extension must precede unrelated project packages. The selected
    # hy3dshape package remains independently importable.
    paths = (
        hunyuan_root / "hy3dpaint" / "custom_rasterizer",
        hunyuan_root / "hy3dpaint",
        *shape_paths,
        hunyuan_root,
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def patch_omni_dino_loader(dino_path: Path) -> None:
    """Resolve Omni's DINO encoder from the configured local snapshot."""
    from transformers import Dinov2Model

    original_from_pretrained = Dinov2Model.from_pretrained

    def local_from_pretrained(_cls, _model_name, *args, **kwargs):
        kwargs["local_files_only"] = True
        return original_from_pretrained(str(dino_path), *args, **kwargs)

    Dinov2Model.from_pretrained = classmethod(local_from_pretrained)


def load_omni_point_cloud(path: Path, pipeline):
    """Load renderer Z-up points, convert to Omni coordinates and normalize."""
    import numpy as np
    import torch
    import trimesh

    point_geometry = trimesh.load(path, process=False)
    if isinstance(point_geometry, trimesh.Scene):
        geometries = tuple(point_geometry.geometry.values())
        if not geometries:
            raise ValueError(f"Point cloud contains no geometry: {path}")
        points = np.concatenate(
            [np.asarray(geometry.vertices) for geometry in geometries], axis=0
        )
    else:
        points = np.asarray(point_geometry.vertices)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Expected a non-empty Nx3 point cloud: {path}")

    points = np.array(points, dtype=np.float32, order="C", copy=True)
    # MajutsuCity/PyVista renders in Z-up coordinates; the original Omni
    # integration exchanged Y and Z before sending the control points.
    points[:, [1, 2]] = points[:, [2, 1]]
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    extent = float((upper - lower).max())
    if extent <= 0:
        raise ValueError(f"Point cloud has zero spatial extent: {path}")
    points = (points - (lower + upper) / 2.0) * (1.96 / extent)
    return torch.from_numpy(points).unsqueeze(0).to(
        device=pipeline.device,
        dtype=pipeline.dtype,
    )


def run_shape(args: argparse.Namespace, jobs: list[dict]) -> None:
    import torch
    from PIL import Image

    pending_jobs = [
        paths
        for paths in jobs
        if not may_reuse_shape(args, paths)
    ]
    if not pending_jobs:
        print("[shape] All selected untextured GLBs already exist; skipping model load.")
        return

    if args.shape_backend == "omni":
        patch_omni_dino_loader(pending_jobs[0]["omni_dino_path"])
        from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
        from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

        print("[shape] Loading local Hunyuan3D-Omni shape model", flush=True)
        pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
            str(pending_jobs[0]["omni_model_path"])
        )
    else:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        print("[shape] Loading local Hunyuan3D-2.1 shape model", flush=True)
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(pending_jobs[0]["model_path"])
        )
    for job_index, paths in enumerate(pending_jobs, start=1):
        building_id = paths["building_id"]
        asset_name = paths.get("asset_name", f"building_{building_id}")
        style_prefix = f"{paths['style']}/" if paths.get("style") else ""
        image = (
            str(paths["input_image"])
            if args.shape_backend == "omni"
            else Image.open(paths["input_image"]).convert("RGBA")
        )
        generator = torch.Generator(device="cuda").manual_seed(args.seed + building_id)
        print(
            f"[shape {job_index}/{len(pending_jobs)}] Generating "
            f"{style_prefix}{asset_name}",
            flush=True,
        )
        if args.shape_backend == "omni":
            points = load_omni_point_cloud(paths["point_cloud"], pipeline)
            result = pipeline(
                image=image,
                point=points,
                num_inference_steps=args.shape_steps,
                guidance_scale=args.guidance_scale,
                octree_resolution=args.octree_resolution,
                mc_level=0,
                generator=generator,
            )
            mesh = result["shapes"][0][0]
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            del points
            del result
        else:
            mesh = pipeline(
                image=image,
                num_inference_steps=args.shape_steps,
                guidance_scale=args.guidance_scale,
                octree_resolution=args.octree_resolution,
                generator=generator,
            )[0]
        mesh.export(paths["untextured_glb"])
        write_record(
            record_path(paths, "shape"),
            expected_shape_record(args, paths),
        )
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
        if not may_reuse_paint(args, paths)
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
        write_record(
            record_path(paths, "paint"),
            expected_paint_record(args, paths),
        )
        print(f"[paint] Saved: {paths['textured_glb']}", flush=True)

    del pipeline
    release_cuda_memory()


def print_summary(args: argparse.Namespace, jobs: list[dict]) -> None:
    print("Resolved Hunyuan3D job:")
    print(f"  stage:          {args.stage}")
    print(f"  shape backend:  {args.shape_backend}")
    print(f"  physical GPU:   {args.gpu}")
    if args.shape_backend == "omni" and args.stage in ("shape", "all"):
        print(f"  Omni model:     {jobs[0]['omni_model_path']}")
        print(f"  Omni DINO:      {jobs[0]['omni_dino_path']}")
    if args.stage in ("paint", "all") or args.shape_backend == "2.1":
        print(f"  2.1 model:      {jobs[0]['model_path']}")
    print(f"  building count: {len(jobs)}")
    print(f"  building IDs:   {[job['building_id'] for job in jobs]}")
    for paths in jobs:
        style_prefix = f"{paths['style']}/" if paths.get("style") else ""
        controls = str(paths["input_image"])
        if args.shape_backend == "omni" and args.stage in ("shape", "all"):
            controls += f" + {paths['point_cloud']}"
        print(
            f"  {style_prefix}building_{paths['building_id']}: "
            f"{controls} -> {paths['textured_glb']}"
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
        if (
            args.input_image is not None
            or args.point_cloud is not None
            or args.output_dir is not None
        ):
            raise ValueError(
                "--input-image, --point-cloud and --output-dir cannot be "
                "combined with --batch-job-file."
            )
        jobs = load_batch_jobs(args)
    else:
        building_ids = parse_building_id_spec(args)
        if len(building_ids) > 1 and (
            args.input_image is not None
            or args.point_cloud is not None
            or args.output_dir is not None
        ):
            raise ValueError(
                "--input-image, --point-cloud and --output-dir can only be "
                "used with a single building."
            )
        jobs = [resolve_paths(args, building_id) for building_id in building_ids]
    for paths in jobs:
        validate_paths(args, paths)
    print_summary(args, jobs)
    if args.dry_run:
        print("Dry run passed; no model was loaded.")
        return

    configure_runtime(args.gpu)
    configure_import_paths(
        args.shape_backend,
        jobs[0]["omni_root"],
        jobs[0]["hunyuan_root"],
    )
    for paths in jobs:
        paths["output_dir"].mkdir(parents=True, exist_ok=True)

    if args.stage in ("shape", "all"):
        run_shape(args, jobs)
    if args.stage in ("paint", "all"):
        run_paint(args, jobs)
        silence_blender_shutdown_memory_report()


if __name__ == "__main__":
    main()
