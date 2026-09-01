#!/usr/bin/env python3
"""Render layout buildings and edit them with a local Qwen-Image-Edit model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from _runtime import env_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNCITY_PYTHON = env_path(
    "MAJUTSUCITY_RENDER_PYTHON", "envs/majutsucity/bin/python"
)
QWEN_PYTHON = env_path(
    "MAJUTSUCITY_QWEN_PYTHON", "envs/majutsucity/bin/python"
)
DEFAULT_QWEN_MODEL = env_path(
    "MAJUTSUCITY_QWEN_IMAGE_EDIT_MODEL", "models/Qwen-Image-Edit-2509"
)
DEFAULT_BACKGROUND_MODEL = env_path(
    "MAJUTSUCITY_RMBG_MODEL", "models/RMBG-2.0"
)
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "image_prompts" / "mc.json"
DEFAULT_LAYOUT = PROJECT_ROOT / "examples/pair_cases/layout/SE-Stockholm_64.png"
DEFAULT_DEPTH = PROJECT_ROOT / "examples/pair_cases/depth/SE-Stockholm_64.png"

EDIT_CONSTRAINT = (
    "Edit the input gray building only. Preserve its silhouette, overall "
    "proportions, footprint, height impression, orthographic isometric camera "
    "angle, framing, scale, and composition. Replace the gray surfaces with "
    "the architecture and materials described below. Keep exactly one complete "
    "building centered on a pure white background. Do not add ground, terrain, "
    "roads, vegetation, people, vehicles, surrounding objects, text, logos, "
    "background shadows, or a base/platform."
)

DEFAULT_NEGATIVE_PROMPT = (
    "changed silhouette, changed perspective, changed camera angle, cropped "
    "building, multiple buildings, ground, terrain, road, vegetation, people, "
    "vehicles, surrounding objects, base, platform, text, logo, watermark, sky, "
    "night, fog, heavy shadow, wide-angle distortion"
)

SCENE_PLAN_BUILDING_REQUIREMENT = (
    "Apply the Scene Design appearance to the input gray building while keeping "
    "the existing isometric camera angle and composition. Show exactly one "
    "complete building on a pure white background, without ground, terrain, a "
    "base, surrounding objects, text, logos or background shadows."
)

LEGACY_CASE_PROMPTS = {
    "christmas": (
        "Edit the input image only; keep the existing camera angle and composition. "
        "Convert the building into a single richly detailed festive Christmas-style "
        "building with gingerbread, warm wood, red velvet, gold trim, snow, wreaths, "
        "ornaments and colorful lights, isolated on a white background."
    ),
    "cyberpunk": (
        "A highly detailed ISOMETRIC building designed in the style of a Cyberpunk "
        "game, with dark futuristic architecture, luminous cyan and magenta neon, "
        "holographic facade details, and a white background."
    ),
    "european": (
        "An isometric building matching the reference massing, designed in a richly "
        "detailed European classical architectural style with masonry, decorative "
        "cornices, framed windows and realistic materials, isolated on a white background."
    ),
    "genshin": (
        "A highly detailed ISOMETRIC fantasy building designed in the colorful, ornate "
        "anime game style of Genshin Impact, with elegant carved structure, luminous "
        "accents and layered roof details, isolated on a white background."
    ),
    "ghibli": (
        "A highly detailed ISOMETRIC building designed in the warm whimsical style of "
        "a Ghibli anime village, with hand-painted materials and charming architectural "
        "details, isolated on a white background."
    ),
    "halloween": (
        "Edit the input image only; keep the existing camera angle and composition. "
        "Convert the building into a richly detailed spooky Halloween-style haunted "
        "building with weathered wood, crumbling stone, wrought iron, pumpkins, webs "
        "and eerie glowing windows, isolated on a white background."
    ),
    "mc": (
        "An ISOMETRIC building with richly detailed Minecraft pixel-art and voxel style, "
        "matching the reference massing, isolated on a plain white background."
    ),
    "modern": (
        "Edit the input image only; keep the existing isometric camera angle and "
        "composition. Convert the building into a highly detailed realistic modern "
        "urban building with clear glass curtain walls, structural steel, clean "
        "rectilinear geometry, concrete or stone panels, realistic windows, and "
        "restrained contemporary materials. Keep exactly one complete building "
        "isolated on a pure white background; do not add plazas, roads, terrain, "
        "vegetation, vehicles, people, or surrounding buildings."
    ),
    "springfest": (
        "Edit the input image only; keep the existing camera angle and composition. "
        "Convert the building into a richly detailed Chinese Spring Festival building "
        "with red silk, polished gold, carved wood, lanterns, paper-cut motifs and jade "
        "accents, isolated on a white background."
    ),
    "technical": (
        "Edit the input image only; keep the existing camera angle and composition. "
        "Convert the building into a highly detailed bright futuristic technical-city "
        "building with white composite panels, satin aluminum, blue glass, louvers, "
        "panel seams and luminous accents, isolated on a white background."
    ),
    "vangogh": (
        "Edit the input image only; keep the existing camera angle and composition. "
        "Convert the building into a richly detailed architectural design rendered in "
        "the style of a Van Gogh painting, with cobalt blue, sunflower yellow, swirling "
        "impasto brushwork and energetic outlines, isolated on a white background."
    ),
}

LEGACY_BUILDING_ROLES = (
    "restaurant",
    "cafe",
    "supermarket",
    "apartment",
    "hospital",
    "bakery",
)

MODERN_BUILDING_ROLES = (
    "mixed-use residential and retail block",
    "glass office tower",
    "contemporary apartment building",
    "commercial center",
    "civic administration building",
    "business hotel",
    "modern hospital",
    "public library",
    "urban school",
    "transportation hub",
    "cultural center",
    "street-front retail pavilion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use layout-derived gray renders and per-building JSON prompts to "
            "run local Qwen Image Edit."
        )
    )
    parser.add_argument(
        "--building-ids",
        default="28",
        help="Comma-separated IDs, a range such as 0-41, or 'all' (default: 28).",
    )
    parser.add_argument(
        "--case-name",
        default="SE-Stockholm_64",
        help="Output case directory name.",
    )
    parser.add_argument(
        "--assets-name",
        default="assets",
        help="Asset directory under output/<case-name> (default: assets).",
    )
    parser.add_argument(
        "--layout-path",
        type=Path,
        default=DEFAULT_LAYOUT,
        help="Color-coded layout image.",
    )
    parser.add_argument(
        "--depth-path",
        type=Path,
        default=DEFAULT_DEPTH,
        help="Layout height image.",
    )
    parser.add_argument(
        "--layout-palette",
        choices=("citydreamer", "majutsu", "auto"),
        default="citydreamer",
        help="Semantic palette used by the layout.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="JSON list containing one design prompt per prompt index.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_QWEN_MODEL,
        help="Local Qwen-Image-Edit-2509 directory.",
    )
    parser.add_argument(
        "--background-model-path",
        type=Path,
        default=DEFAULT_BACKGROUND_MODEL,
        help="Local RMBG-2.0 directory.",
    )
    parser.add_argument(
        "--gpu",
        default="0,1",
        help=(
            "Exactly two physical GPUs used by Qwen balanced loading, "
            "for example 0,1 or 4,5."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shape-backend",
        choices=("omni", "2.1"),
        default="omni",
        help="Downstream shape backend; Omni requires sampled point clouds.",
    )
    parser.add_argument(
        "--point-samples",
        type=int,
        default=50,
        help=(
            "Number of surface points saved for Hunyuan3D-Omni point control "
            "for each building (default: 50)."
        ),
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument(
        "--edit-profile",
        choices=("legacy", "constrained"),
        default="legacy",
        help=(
            "legacy uses one case-level style prompt and permits rich architectural "
            "redesign; constrained maps building_i to prompt_i, cycles the prompt "
            "list when the layout has more buildings, and strongly preserves the "
            "gray massing (default: legacy)."
        ),
    )
    parser.add_argument(
        "--case-prompt",
        default=None,
        help=(
            "Scene-level building description. It overrides the legacy case prompt "
            "and refines each per-building prompt in constrained mode."
        ),
    )
    parser.add_argument(
        "--enable-vlm-review",
        action="store_true",
        help="Review edited building images through an OpenAI-compatible vision API.",
    )
    parser.add_argument(
        "--vlm-max-attempts",
        type=int,
        default=3,
        help="Maximum edit attempts for each VLM-reviewed building.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help=(
            "Negative prompt passed to Qwen Image Edit. Defaults to a blank prompt "
            "for legacy and the strict exclusion list for constrained."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate and overwrite existing RGB/edit/background outputs.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Reuse existing rgb.png/depth.png files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and prompt mappings without rendering or loading models.",
    )
    parser.add_argument(
        "--batch-job-file",
        type=Path,
        default=None,
        help=(
            "JSON job list for processing multiple style directories while "
            "loading Qwen and RMBG only once."
        ),
    )
    parser.add_argument(
        "--worker",
        choices=("render", "qwen", "render-batch", "qwen-batch"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--resolved-building-ids", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_prompts(prompt_file: Path) -> list[str]:
    with prompt_file.open("r", encoding="utf-8") as file:
        prompts = json.load(file)
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"Expected a JSON list of strings: {prompt_file}")
    return prompts


def layout_building_ids(layout_path: Path, palette_name: str = "auto") -> list[int]:
    import numpy as np
    from PIL import Image

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils.buildings_extract import extract_objects_info
    from utils.function import image_to_mask

    image = Image.open(layout_path).convert("RGB")
    layout_mask = image_to_mask(image, palette_name=palette_name)
    _, _, objects = extract_objects_info(
        "building",
        layout_mask,
        np.zeros((image.height, image.width), dtype=np.float32),
    )
    return [int(object_info.id) for object_info in objects]


def count_layout_buildings(layout_path: Path, palette_name: str = "auto") -> int:
    return len(layout_building_ids(layout_path, palette_name=palette_name))


def parse_id_spec(
    spec: str,
    building_count: int,
    valid_ids: Iterable[int] | None = None,
) -> list[int]:
    available_ids = (
        set(range(building_count)) if valid_ids is None else set(valid_ids)
    )
    if spec.strip().lower() == "all":
        return sorted(available_ids)

    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending building range: {part}")
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))

    result = sorted(ids)
    invalid = [building_id for building_id in result if building_id not in available_ids]
    if invalid:
        raise ValueError(
            f"Building IDs unavailable in the extracted layout: {invalid}"
        )
    if not result:
        raise ValueError("No building IDs selected.")
    return result


def select_prompt(building_id: int, prompts: list[str]) -> tuple[int, str]:
    if not prompts:
        raise ValueError("At least one building prompt is required.")

    # Preserve building_i -> prompts[i] for the first prompt_count IDs. Larger
    # generated layouts reuse the ordered style vocabulary deterministically
    # instead of failing after model loading.
    prompt_index = building_id % len(prompts)
    #
    # TODO: Replace ID mapping with geometry-aware prompt selection. Use layout
    # height, footprint circularity, area, and aspect ratio to match prompt
    # massing constraints such as low/high-rise, circular/rectangular, and
    # narrow/wide before applying Qwen Image Edit.
    return prompt_index, prompts[prompt_index]


def constrained_edit_prompt(design_prompt: str) -> str:
    return f"{EDIT_CONSTRAINT}\n\nDesign prompt:\n{design_prompt}"


def style_from_assets_name(assets_name: str) -> str | None:
    if assets_name.startswith("assets_") and len(assets_name) > len("assets_"):
        return assets_name.removeprefix("assets_")
    return None


def legacy_case_prompt(style: str | None, override: str | None) -> str:
    if override:
        return (
            f"Scene Design description:\n{override}\n\n"
            f"Mandatory generation requirements:\n{SCENE_PLAN_BUILDING_REQUIREMENT}"
        )
    if style in LEGACY_CASE_PROMPTS:
        return LEGACY_CASE_PROMPTS[style]
    style_text = (style or "stylized").replace("_", " ")
    return (
        f"A highly detailed ISOMETRIC single building designed in a rich "
        f"{style_text} style, with elaborate architectural materials and facade "
        f"details, isolated on a white background."
    )


def resolve_edit_prompt(
    args: argparse.Namespace,
    building_id: int,
    prompts: list[str],
    style: str | None,
) -> tuple[int, str, str, str]:
    if args.edit_profile == "legacy":
        design_prompt = legacy_case_prompt(style, args.case_prompt)
        # The original case pipeline varied these six building roles while
        # keeping one broad case-level style prompt. Use a deterministic role
        # per ID so a resumed batch reproduces the same image.
        roles = (
            MODERN_BUILDING_ROLES
            if style == "modern"
            else LEGACY_BUILDING_ROLES
        )
        role = roles[building_id % len(roles)]
        final_prompt = f"{design_prompt} The building is a {role}."
        return (
            -1,
            design_prompt,
            final_prompt,
            "case_style_prompt_with_legacy_role",
        )

    prompt_index, design_prompt = select_prompt(building_id, prompts)
    if args.case_prompt:
        design_prompt = (
            f"Scene Design description:\n{args.case_prompt}\n\n"
            f"Building-specific design variation:\n{design_prompt}"
        )
    mapping = (
        "building_id_to_prompt_index"
        if building_id < len(prompts)
        else "building_id_mod_prompt_count"
    )
    return (
        prompt_index,
        design_prompt,
        constrained_edit_prompt(design_prompt),
        mapping,
    )


def validate_assets_name(assets_name: str) -> None:
    if (
        not assets_name
        or assets_name in (".", "..")
        or Path(assets_name).name != assets_name
    ):
        raise ValueError(
            "--assets-name must be one directory name, for example assets_mc"
        )


def building_dir(case_name: str, assets_name: str, building_id: int) -> Path:
    return (
        PROJECT_ROOT
        / "output"
        / case_name
        / assets_name
        / "buildings"
        / f"building_{building_id}"
    )


def required_render_names(shape_backend: str) -> tuple[str, ...]:
    names = ("rgb.png", "depth.png")
    return (*names, "points_ascii.ply") if shape_backend == "omni" else names


def validate_gpu_spec(gpu_spec: str) -> None:
    gpu_ids = [item.strip() for item in gpu_spec.split(",") if item.strip()]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("--gpu must contain exactly two distinct physical GPUs.")
    invalid = [gpu_id for gpu_id in gpu_ids if not gpu_id.isdecimal()]
    if invalid:
        raise ValueError(
            "--gpu only accepts non-negative physical GPU IDs; invalid values: "
            + ", ".join(invalid)
        )


def validate_common_paths(args: argparse.Namespace) -> None:
    required = {
        "layout": args.layout_path,
        "depth": args.depth_path,
        "Qwen model": args.model_path,
        "background model": args.background_model_path,
        "render Python": SYNCITY_PYTHON,
        "Qwen Python": QWEN_PYTHON,
    }
    if not args.batch_job_file:
        required["prompt file"] = args.prompt_file
    else:
        required["batch job file"] = args.batch_job_file
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))


def resolved_ids_from_args(args: argparse.Namespace) -> list[int]:
    if args.resolved_building_ids:
        return [int(item) for item in args.resolved_building_ids.split(",")]
    valid_ids = layout_building_ids(
        args.layout_path, palette_name=args.layout_palette
    )
    return parse_id_spec(args.building_ids, len(valid_ids), valid_ids=valid_ids)


def render_worker(args: argparse.Namespace, building_ids: Iterable[int]) -> None:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(PROJECT_ROOT))
    from utils.buildings_extract import extract_objects_info
    from utils.function import image_to_mask
    from utils.render_mesh import RenderMesh

    layout = Image.open(args.layout_path).convert("RGB")
    layout_mask = image_to_mask(layout, palette_name=args.layout_palette)
    height_map = np.asarray(Image.open(args.depth_path).convert("L"), dtype=np.float32)
    _, _, objects = extract_objects_info("building", layout_mask, height_map)
    object_by_id = {object_info.id: object_info for object_info in objects}

    renderer = RenderMesh()
    renderer.setup_offscreen_rendering()
    for building_id in building_ids:
        if building_id not in object_by_id:
            raise KeyError(f"Layout extraction did not produce building_{building_id}")
        output_dir = building_dir(args.case_name, args.assets_name, building_id)
        rgb_path = output_dir / "rgb.png"
        depth_path = output_dir / "depth.png"
        point_path = output_dir / "points_ascii.ply"
        render_is_complete = (
            not args.overwrite
            and rgb_path.is_file()
            and depth_path.is_file()
            and (args.shape_backend == "2.1" or point_path.is_file())
        )
        if render_is_complete:
            print(f"[render] Keeping existing building_{building_id}", flush=True)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        object_info = object_by_id[building_id]
        rendered = renderer.build_and_render_from_mask(
            mask=object_info.mask.astype(bool),
            height=object_info.height,
            pixel_size=0.25,
            simplify_tol=0,
            min_area_px=40,
            obj_info=object_info,
            export_osm=True,
        )
        renderer.save_render_image(rendered, str(output_dir))
        if args.shape_backend == "omni":
            point_cloud = renderer.sample_points_from_mesh(
                rendered["mesh"],
                num_samples=args.point_samples,
                with_normals=False,
                method="barycentric",
                seed=args.seed + building_id,
            )
            point_cloud.clear_data()
            point_cloud.save(point_path, binary=False)
            products = "rgb.png, depth.png and points_ascii.ply"
        else:
            products = "rgb.png and depth.png"
        print(f"[render] Saved building_{building_id}/{products}", flush=True)


def write_prompt_record(
    output_dir: Path,
    building_id: int,
    prompt_index: int,
    design_prompt: str,
    final_prompt: str,
    args: argparse.Namespace,
    style: str | None = None,
    prompt_file: Path | None = None,
    mapping: str = "building_id_to_prompt_index",
) -> None:
    record = {
        "building_id": building_id,
        "mapping": mapping,
        "edit_profile": args.edit_profile,
        "prompt_index": prompt_index,
        "design_prompt": design_prompt,
        "edit_prompt": final_prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed + building_id,
        "model_path": str(args.model_path),
        "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
    }
    if "actual_seed" in args.__dict__:
        record["seed"] = args.actual_seed
    if args.enable_vlm_review and "review_attempts" in args.__dict__:
        record["vlm_review"] = {
            "enabled": True,
            "kind": "building",
            "attempts": args.review_attempts,
            "accepted": True,
        }
    if style is not None:
        record["style"] = style
    if prompt_file is not None:
        record["prompt_file"] = str(prompt_file)
    with (output_dir / "prompt.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def collect_qwen_jobs(
    args: argparse.Namespace,
    building_ids: Iterable[int],
    *,
    style: str | None = None,
) -> tuple[list[dict], list[dict]]:
    prompts = load_prompts(args.prompt_file)
    resolved_style = style or style_from_assets_name(args.assets_name)
    edit_jobs = []
    background_jobs = []

    for building_id in building_ids:
        output_dir = building_dir(args.case_name, args.assets_name, building_id)
        rgb_path = output_dir / "rgb.png"
        image_path = output_dir / "image.png"
        no_background_path = output_dir / "image_woBG.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing gray render: {rgb_path}")
        if no_background_path.is_file() and not args.overwrite:
            print(f"[qwen] Keeping existing building_{building_id}", flush=True)
            continue

        prompt_index, design_prompt, final_prompt, mapping = resolve_edit_prompt(
            args,
            building_id,
            prompts,
            resolved_style,
        )
        job = {
            "style": resolved_style,
            "prompt_file": args.prompt_file,
            "building_id": building_id,
            "output_dir": output_dir,
            "rgb_path": rgb_path,
            "image_path": image_path,
            "no_background_path": no_background_path,
            "prompt_index": prompt_index,
            "design_prompt": design_prompt,
            "final_prompt": final_prompt,
            "mapping": mapping,
            "negative_prompt": args.negative_prompt,
        }
        if args.overwrite or not image_path.is_file():
            edit_jobs.append(job)
        else:
            background_jobs.append(job)
    return edit_jobs, background_jobs


def execute_qwen_jobs(
    args: argparse.Namespace,
    edit_jobs: list[dict],
    background_jobs: list[dict],
) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from PIL import Image

    from utils.qwen_image import Qwen_Image
    from utils.rembg import BackgroundRemover
    from utils.openai_vlm import OpenAICompatibleVLM

    reviewer = (
        OpenAICompatibleVLM.from_environment()
        if args.enable_vlm_review
        else None
    )

    if reviewer is not None and background_jobs:
        accepted_existing = []
        for job in background_jobs:
            accepted = reviewer.image_review(
                job["image_path"], review_kind="building"
            )
            print(
                f"[VLM review] existing building_{job['building_id']}: {accepted}",
                flush=True,
            )
            if accepted:
                accepted_existing.append(job)
            else:
                edit_jobs.append(job)
        background_jobs = accepted_existing

    qwen = None
    if edit_jobs:
        print(f"[qwen] Loading local model for {len(edit_jobs)} edit job(s)", flush=True)
        qwen = Qwen_Image(
            model_name=str(args.model_path),
            device="cuda",
            local_files_only=True,
            device_map="balanced",
        )
        for job in edit_jobs:
            building_id = job["building_id"]
            style_prefix = f"{job['style']}/" if job["style"] else ""
            print(
                f"[qwen] {style_prefix}building_{building_id} -> "
                f"prompt_{job['prompt_index']}: "
                f"{job['design_prompt']}",
                flush=True,
            )
            for attempt in range(args.vlm_max_attempts if reviewer else 1):
                actual_seed = args.seed + building_id + attempt
                image = qwen.generate(
                    prompt=job["final_prompt"],
                    negative_prompt=args.negative_prompt,
                    seed=actual_seed,
                    image=Image.open(job["rgb_path"]).convert("RGB"),
                    num_inference_steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    memory_optimize=False,
                )
                image.save(job["image_path"])
                if reviewer is None:
                    break
                accepted = reviewer.image_review(
                    job["image_path"], review_kind="building"
                )
                print(
                    f"[VLM review] building_{building_id} attempt "
                    f"{attempt + 1}/{args.vlm_max_attempts}: {accepted}",
                    flush=True,
                )
                if accepted:
                    break
            else:
                raise RuntimeError(
                    f"VLM review rejected building_{building_id} after "
                    f"{args.vlm_max_attempts} attempt(s)."
                )
            args.actual_seed = actual_seed
            args.review_attempts = attempt + 1
            write_prompt_record(
                job["output_dir"],
                building_id,
                job["prompt_index"],
                job["design_prompt"],
                job["final_prompt"],
                args,
                style=job["style"],
                prompt_file=job["prompt_file"],
                mapping=job["mapping"],
            )
            background_jobs.append(job)
            print(f"[qwen] Saved: {job['image_path']}", flush=True)

        qwen.unload()
        del qwen

    if background_jobs:
        print("[rembg] Loading local background-removal model", flush=True)
        remover = BackgroundRemover(
            model_path=str(args.background_model_path),
            device="cuda",
            local_files_only=True,
        )
        for job in background_jobs:
            result = remover(Image.open(job["image_path"]).convert("RGB"))
            result.save(job["no_background_path"])
            print(f"[rembg] Saved: {job['no_background_path']}", flush=True)


def qwen_worker(args: argparse.Namespace, building_ids: Iterable[int]) -> None:
    edit_jobs, background_jobs = collect_qwen_jobs(args, building_ids)
    execute_qwen_jobs(args, edit_jobs, background_jobs)


def load_batch_specs(batch_job_file: Path) -> list[dict]:
    with batch_job_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    raw_jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError(f"Expected a non-empty 'jobs' list: {batch_job_file}")

    specs = []
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ValueError(f"Invalid batch job {index}: expected an object.")
        style = raw_job.get("style")
        case_name = raw_job.get("case_name")
        assets_name = raw_job.get("assets_name")
        prompt_file = Path(str(raw_job.get("prompt_file", ""))).expanduser().resolve()
        building_ids = raw_job.get("building_ids")
        if not isinstance(style, str) or not style:
            raise ValueError(f"Invalid style in batch job {index}.")
        if not isinstance(case_name, str) or not case_name:
            raise ValueError(f"Invalid case_name in batch job {index}.")
        if not isinstance(assets_name, str):
            raise ValueError(f"Invalid assets_name in batch job {index}.")
        validate_assets_name(assets_name)
        if not prompt_file.is_file():
            raise FileNotFoundError(f"Missing prompt file: {prompt_file}")
        if (
            not isinstance(building_ids, list)
            or not building_ids
            or not all(isinstance(item, int) for item in building_ids)
        ):
            raise ValueError(f"Invalid building_ids in batch job {index}.")
        specs.append(
            {
                "style": style,
                "case_name": case_name,
                "assets_name": assets_name,
                "prompt_file": prompt_file,
                "building_ids": building_ids,
            }
        )
    return specs


def args_for_batch_spec(
    args: argparse.Namespace,
    spec: dict,
) -> argparse.Namespace:
    job_args = argparse.Namespace(**vars(args))
    job_args.case_name = spec["case_name"]
    job_args.assets_name = spec["assets_name"]
    job_args.prompt_file = spec["prompt_file"]
    return job_args


def render_batch_worker(args: argparse.Namespace, specs: list[dict]) -> None:
    for spec in specs:
        print(f"[render] Style: {spec['style']}", flush=True)
        render_worker(args_for_batch_spec(args, spec), spec["building_ids"])


def qwen_batch_worker(args: argparse.Namespace, specs: list[dict]) -> None:
    all_edit_jobs = []
    all_background_jobs = []
    for spec in specs:
        job_args = args_for_batch_spec(args, spec)
        edit_jobs, background_jobs = collect_qwen_jobs(
            job_args,
            spec["building_ids"],
            style=spec["style"],
        )
        all_edit_jobs.extend(edit_jobs)
        all_background_jobs.extend(background_jobs)

    print(
        f"[qwen-batch] Pending edits: {len(all_edit_jobs)}; "
        f"pending background removals: {len(all_background_jobs)}",
        flush=True,
    )
    execute_qwen_jobs(args, all_edit_jobs, all_background_jobs)


def worker_command(args: argparse.Namespace, worker: str, building_ids: list[int]) -> list[str]:
    interpreter = SYNCITY_PYTHON if worker == "render" else QWEN_PYTHON
    original_args = [
        argument
        for argument in sys.argv[1:]
        if argument not in ("--dry-run",)
    ]
    return [
        str(interpreter),
        str(Path(__file__).resolve()),
        *original_args,
        "--worker",
        worker,
        "--resolved-building-ids",
        ",".join(str(building_id) for building_id in building_ids),
    ]


def batch_worker_command(args: argparse.Namespace, worker: str) -> list[str]:
    interpreter = SYNCITY_PYTHON if worker == "render-batch" else QWEN_PYTHON
    original_args = [
        argument
        for argument in sys.argv[1:]
        if argument not in ("--dry-run",)
    ]
    return [
        str(interpreter),
        str(Path(__file__).resolve()),
        *original_args,
        "--worker",
        worker,
    ]
def parent_main(args: argparse.Namespace) -> None:
    if args.vlm_max_attempts <= 0:
        raise ValueError("--vlm-max-attempts must be positive.")
    if args.point_samples <= 0:
        raise ValueError("--point-samples must be positive.")
    validate_common_paths(args)
    validate_gpu_spec(args.gpu)
    valid_ids = layout_building_ids(
        args.layout_path, palette_name=args.layout_palette
    )
    building_count = len(valid_ids)
    building_ids = parse_id_spec(
        args.building_ids, building_count, valid_ids=valid_ids
    )
    prompts = load_prompts(args.prompt_file)

    print("Resolved Qwen Image Edit job:")
    print(f"  buildings:       {building_ids}")
    print(f"  layout count:    {building_count}")
    print(f"  prompt count:    {len(prompts)}")
    print(f"  physical GPUs:   {args.gpu}")
    print(f"  edit profile:    {args.edit_profile}")
    print(f"  overwrite:       {args.overwrite}")
    for building_id in building_ids:
        prompt_index, _, prompt, mapping = resolve_edit_prompt(
            args,
            building_id,
            prompts,
            style_from_assets_name(args.assets_name),
        )
        print(
            f"  building_{building_id} -> {mapping}"
            f"{'' if prompt_index < 0 else f'_{prompt_index}'}: {prompt}"
        )

    if args.skip_render:
        missing_render_assets = [
            path
            for building_id in building_ids
            for name in required_render_names(args.shape_backend)
            for path in (
                building_dir(args.case_name, args.assets_name, building_id) / name,
            )
            if not path.is_file()
        ]
        if missing_render_assets:
            raise FileNotFoundError(
                "--skip-render requested but render/point-cloud assets are missing:\n  "
                + "\n  ".join(str(path) for path in missing_render_assets)
            )

    if args.dry_run:
        print("Dry run passed; no renderer or model was loaded.")
        return

    if not args.skip_render:
        render_env = os.environ.copy()
        render_env["MPLCONFIGDIR"] = "/tmp/majutsu_qwen_mpl"
        subprocess.run(
            worker_command(args, "render", building_ids),
            cwd=PROJECT_ROOT,
            env=render_env,
            check=True,
        )

    qwen_env = os.environ.copy()
    qwen_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    qwen_env["HF_HUB_OFFLINE"] = "1"
    qwen_env["TRANSFORMERS_OFFLINE"] = "1"
    qwen_env["DIFFUSERS_OFFLINE"] = "1"
    qwen_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(
        worker_command(args, "qwen", building_ids),
        cwd=PROJECT_ROOT,
        env=qwen_env,
        check=True,
    )

    print("Qwen image editing completed.")
    print("Any existing building.glb files for overwritten IDs are now stale.")


def batch_parent_main(args: argparse.Namespace) -> None:
    validate_common_paths(args)
    validate_gpu_spec(args.gpu)
    specs = load_batch_specs(args.batch_job_file)
    valid_ids = layout_building_ids(
        args.layout_path, palette_name=args.layout_palette
    )

    print("Resolved persistent Qwen batch:")
    print(f"  physical GPUs: {args.gpu}")
    print(f"  style jobs:    {len(specs)}")
    for spec in specs:
        prompts = load_prompts(spec["prompt_file"])
        invalid = [
            building_id
            for building_id in spec["building_ids"]
            if building_id not in valid_ids
        ]
        if invalid:
            raise ValueError(
                f"{spec['style']} has invalid building IDs for layout: {invalid}"
            )
        for building_id in spec["building_ids"]:
            select_prompt(building_id, prompts)
        print(
            f"  {spec['style']}: {len(spec['building_ids'])} buildings, "
            f"{len(prompts)} prompts -> {spec['assets_name']}"
        )

    if args.skip_render:
        missing_render_assets = []
        for spec in specs:
            for building_id in spec["building_ids"]:
                directory = (
                    building_dir(
                        spec["case_name"],
                        spec["assets_name"],
                        building_id,
                    )
                )
                for name in required_render_names(args.shape_backend):
                    path = directory / name
                    if not path.is_file():
                        missing_render_assets.append(path)
        if missing_render_assets:
            raise FileNotFoundError(
                "--skip-render requested but render/point-cloud assets are missing:\n  "
                + "\n  ".join(str(path) for path in missing_render_assets)
            )

    if args.dry_run:
        print("Persistent Qwen batch dry run passed; no model was loaded.")
        return

    if not args.skip_render:
        render_env = os.environ.copy()
        render_env["MPLCONFIGDIR"] = f"/tmp/majutsu_qwen_mpl_{args.gpu.replace(',', '_')}"
        subprocess.run(
            batch_worker_command(args, "render-batch"),
            cwd=PROJECT_ROOT,
            env=render_env,
            check=True,
        )

    qwen_env = os.environ.copy()
    qwen_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    qwen_env["HF_HUB_OFFLINE"] = "1"
    qwen_env["TRANSFORMERS_OFFLINE"] = "1"
    qwen_env["DIFFUSERS_OFFLINE"] = "1"
    qwen_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(
        batch_worker_command(args, "qwen-batch"),
        cwd=PROJECT_ROOT,
        env=qwen_env,
        check=True,
    )
    print("Persistent Qwen batch completed.")


def main() -> None:
    args = parse_args()
    if args.vlm_max_attempts <= 0:
        raise ValueError("--vlm-max-attempts must be positive.")
    if args.point_samples <= 0:
        raise ValueError("--point-samples must be positive.")
    args.layout_path = args.layout_path.expanduser().resolve()
    args.depth_path = args.depth_path.expanduser().resolve()
    args.prompt_file = args.prompt_file.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.background_model_path = args.background_model_path.expanduser().resolve()
    if args.negative_prompt is None:
        args.negative_prompt = (
            " " if args.edit_profile == "legacy" else DEFAULT_NEGATIVE_PROMPT
        )
    if args.batch_job_file:
        args.batch_job_file = args.batch_job_file.expanduser().resolve()
    else:
        validate_assets_name(args.assets_name)

    if args.worker:
        if args.worker == "render":
            building_ids = resolved_ids_from_args(args)
            render_worker(args, building_ids)
        elif args.worker == "qwen":
            building_ids = resolved_ids_from_args(args)
            qwen_worker(args, building_ids)
        else:
            specs = load_batch_specs(args.batch_job_file)
            if args.worker == "render-batch":
                render_batch_worker(args, specs)
            else:
                qwen_batch_worker(args, specs)
        return

    if args.batch_job_file:
        batch_parent_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
