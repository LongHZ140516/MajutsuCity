#!/usr/bin/env python3
"""Generate style-specific trees, lamps, surface textures and skyboxes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from _runtime import env_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.openai_vlm import OpenAICompatibleVLM, load_scene_plan  # noqa: E402
QWEN_PYTHON = env_path(
    "MAJUTSUCITY_QWEN_PYTHON", "envs/majutsucity/bin/python"
)
HUNYUAN_PYTHON = env_path(
    "MAJUTSUCITY_HUNYUAN_PYTHON", "envs/majutsucity/bin/python"
)
HUNYUAN_SCRIPT = PROJECT_ROOT / "scripts" / "run_hunyuan3d_local.py"
TEXTURE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_surface_texture.py"
SKYBOX_SCRIPT = PROJECT_ROOT / "scripts" / "generate_skybox_texture.py"
SURFACE_KINDS = ("ground", "road", "vegetation", "water")
DEFAULT_QWEN_MODEL = env_path(
    "MAJUTSUCITY_QWEN_IMAGE_MODEL", "models/Qwen-Image"
)
DEFAULT_BACKGROUND_MODEL = env_path(
    "MAJUTSUCITY_RMBG_MODEL", "models/RMBG-2.0"
)
DEFAULT_TEXTURE_LORA = env_path(
    "MAJUTSUCITY_TEXTURE_LORA",
    "models/MajutsuCity/texture/Texture_Lora.safetensors",
)
DEFAULT_SKYBOX_LORA = env_path(
    "MAJUTSUCITY_SKYBOX_LORA",
    "models/MajutsuCity/texture/SkyBox_Lora.safetensors",
)

STYLE_DESCRIPTIONS = {
    "christmas": (
        "festive Christmas fantasy, snow-dusted pine foliage, warm red and gold "
        "ornaments, tiny colorful lights"
    ),
    "cyberpunk": (
        "dark cyberpunk visual style, graphite-black materials with restrained "
        "luminous cyan and magenta accents"
    ),
    "european": (
        "realistic European old-town style, natural foliage, carved stone and "
        "wrought-iron detailing"
    ),
    "genshin": (
        "colorful ornate anime fantasy game style, elegant shapes, luminous blue "
        "accents and painterly materials"
    ),
    "ghibli": (
        "warm whimsical hand-painted anime village style, soft natural colors "
        "and charming organic details"
    ),
    "halloween": (
        "spooky Halloween fantasy, twisted silhouettes, weathered dark wood, "
        "pumpkin-orange glow and muted purple accents"
    ),
    "mc": (
        "Minecraft-inspired voxel and pixel-art style, cubic construction, "
        "crisp block textures"
    ),
    "modern": (
        "realistic contemporary metropolitan design, clean restrained forms, "
        "natural greenery, brushed steel, dark graphite and subtle cool-white accents"
    ),
    "springfest": (
        "Chinese Spring Festival style, rich red, polished gold, carved wood, "
        "lantern motifs and jade accents"
    ),
    "technical": (
        "bright futuristic technical-city style, white composite panels, satin "
        "aluminum, blue glass and restrained cyan light"
    ),
    "vangogh": (
        "Van Gogh-inspired painted style, cobalt blue, sunflower yellow, "
        "expressive outlines and swirling impasto texture"
    ),
}

TREE_FORMS = (
    "mature street tree with a compact rounded leafy canopy and a thick visible trunk",
    "slender columnar street tree with a clearly visible trunk and a layered leafy crown",
)

LAMP_FORMS = (
    "a single upright street lamp with one sturdy pole and one compact overhead light",
    "a single upright street lamp with a sturdy pole and two compact symmetric lights",
)

TREE_NEGATIVE = (
    "multiple trees, forest, ground, terrain, planter, pot, platform, people, "
    "building, house, tower, skyscraper, roof, windows, facade, room, street, "
    "vehicle, robot, machine, spherical machine, text, logo, watermark, cropped "
    "crown, cropped trunk, thin disconnected branches, floating leaves, "
    "transparent foliage, shadow"
)

LAMP_NEGATIVE = (
    "multiple lamps, row of lamps, street, road, sidewalk, ground, platform, "
    "pavement, pavement tile, square slab, floor tile, plinth, wide pedestal, "
    "diorama base, building, room, vehicle, people, sign, traffic light, utility "
    "pole, wires, text, logo, watermark, cropped top, cropped base, floating "
    "parts, thin disconnected pieces, transparent body, cast shadow"
)

ROAD_NEGATIVE = (
    "road scene, street scene, perspective, isometric view, horizon, curb, "
    "sidewalk, intersection, edge, vehicles, people, buildings, trees, street "
    "lights, text, arrows, logos, watermark, room, wall, frame, border, vignette, "
    "depth of field, reflections, highlights, directional lighting, cast shadows, "
    "large cracks, paving stones, cobblestone, slabs, tiles, neon, glowing marks, "
    "colored fragments, lane markings, non-tileable border"
)

GROUND_NEGATIVE = (
    "ground scene, street scene, perspective, isometric view, horizon, road, "
    "asphalt, lane markings, curb, sidewalk edge, grass, plants, water, rocks, "
    "buildings, vehicles, people, furniture, text, logo, watermark, room, wall, "
    "frame, border, vignette, reflections, directional lighting, cast shadows, "
    "large cracks, large slabs, strong grid, visible seams"
)

VEGETATION_NEGATIVE = (
    "landscape, garden scene, perspective, isometric view, horizon, soil patch, "
    "road, pavement, water, rocks, trees, trunks, bushes, flowers, buildings, "
    "vehicles, people, text, logo, watermark, frame, border, directional "
    "lighting, cast shadows, large leaves, visible seams"
)

WATER_NEGATIVE = (
    "low quality, blurry, distorted, text, watermark, horizon, perspective, "
    "landscape, aerial view, map, pool, pond, canal, shore, coast, beach, "
    "island, boat, people, objects, road, sidewalk, pavement, curb, wall, "
    "building, vegetation, grass, plants, trees, rocks, frame, border, "
    "directional lighting, cast shadows, visible seams"
)

SKYBOX_NEGATIVE = (
    "low quality, blurry, distorted, text, watermark, frame, border, fisheye, "
    "cube map, ground, terrain, buildings, people, visible vertical seam"
)

ROAD_COLOR_HINTS = {
    "christmas": "neutral cold charcoal gray",
    "cyberpunk": "cool charcoal gray with an extremely faint cyan undertone",
    "european": "neutral weathered asphalt gray",
    "genshin": "soft blue-gray",
    "ghibli": "warm natural charcoal gray",
    "halloween": "muted brown-black",
    "mc": "neutral blocky asphalt gray",
    "modern": "neutral cool asphalt gray",
    "springfest": "warm charcoal gray",
    "technical": "clean cool graphite gray",
    "vangogh": "muted blue-charcoal gray",
}

GROUND_COLOR_HINTS = {
    "christmas": "pale cold-gray urban stone with a restrained snowy tint",
    "cyberpunk": "dark graphite composite pavement with a faint cool undertone",
    "european": "warm weathered beige-gray urban stone",
    "genshin": "soft pale sandstone with restrained blue-gray variation",
    "ghibli": "warm natural beige-gray compacted urban ground",
    "halloween": "dark muted brown-gray weathered stone",
    "mc": "simple muted tan-gray block-surface color",
    "modern": "clean light cool-gray architectural concrete",
    "springfest": "warm neutral gray stone with a faint red-brown undertone",
    "technical": "clean pale blue-white composite pavement",
    "vangogh": "muted ochre and blue-gray painterly ground",
}

VEGETATION_COLOR_HINTS = {
    "christmas": "cold deep green grass with a faint frost tint",
    "cyberpunk": "very dark teal-green urban groundcover",
    "european": "natural muted medium green grass",
    "genshin": "fresh luminous emerald and yellow-green grass",
    "ghibli": "soft hand-painted warm green grass",
    "halloween": "dark olive and desaturated moss-green groundcover",
    "mc": "simple saturated blocky green grass",
    "modern": "natural restrained medium-green maintained grass",
    "springfest": "rich natural green grass with a subtle warm undertone",
    "technical": "clean cool emerald-green maintained grass",
    "vangogh": "expressive olive, emerald and yellow-green painted grass",
}

WATER_COLOR_HINTS = {
    "christmas": "clear icy azure with a faint cool-cyan tint",
    "cyberpunk": "deep navy blue with restrained cyan accents",
    "european": "natural muted blue-green",
    "genshin": "clear luminous turquoise and soft azure",
    "ghibli": "soft hand-painted blue-green",
    "halloween": "dark desaturated blue-violet",
    "mc": "simple saturated medium blue",
    "modern": "clean deep blue-gray with restrained cool-cyan highlights",
    "springfest": "clear deep blue with subtle warm-blue accents",
    "technical": "clean cool cyan-blue",
    "vangogh": "cobalt and ultramarine blue",
}

def tree_prompt(style: str, variant: int) -> str:
    form = TREE_FORMS[variant % len(TREE_FORMS)]
    return (
        f"One complete highly detailed ISOMETRIC {form}. It must be immediately "
        "recognizable as a tree: one trunk, branching limbs, and a foliage crown; "
        "never turn the crown into a building, tower, room, vehicle, or machine. "
        f"Apply {STYLE_DESCRIPTIONS[style]} only to the tree's materials, colors, "
        "surface details, and small decorative accents while preserving the tree "
        "anatomy. Make the foliage crown a connected compact volume suitable for "
        "image-to-3D reconstruction, with all branches and foliage clusters "
        "attached to the trunk. Center exactly one tree on a pure white background. "
        "Show the full crown, trunk, and roots with generous blank margins; no "
        "ground, no base, no surrounding objects, soft uniform studio lighting, "
        "crisp clean silhouette."
    )


def lamp_prompt(style: str, variant: int) -> str:
    form = LAMP_FORMS[variant % len(LAMP_FORMS)]
    return (
        f"One complete highly detailed ISOMETRIC {form}. It must remain clearly "
        "recognizable as a practical street lamp with a continuous load-bearing "
        "pole, a visible ground-contact base, and attached light fixtures. Apply "
        f"{STYLE_DESCRIPTIONS[style]} only to its materials, colors, silhouette, "
        "and restrained decorative details. Keep every component thick, solid, "
        "and connected for image-to-3D reconstruction. The pole must terminate "
        "in a narrow metal mounting foot no wider than twice the pole diameter. "
        "Pure white background must be directly visible on every side of and "
        "immediately beneath that narrow foot: absolutely no pavement tile, "
        "square slab, platform, plinth, pedestal, floor, or diorama base. Center "
        "exactly one lamp on a pure white background, showing the complete top "
        "and mounting foot with generous blank margins. No surrounding objects, "
        "soft uniform studio lighting, crisp clean silhouette."
    )


def road_prompt(style: str) -> str:
    return (
        "A nearly uniform seamless PBR BASE-COLOR / ALBEDO texture of plain fine "
        "asphalt. It must look simple and unmistakably like ordinary asphalt, "
        "with only subtle dense sand-sized grain and tiny nondirectional wear. "
        f"Use {ROAD_COLOR_HINTS[style]}; do not add thematic decorations. Flat "
        "2D material swatch filling the entire square frame, exact 90-degree "
        "top-down orthographic material scan. No large features, cracks, stones, "
        "slabs, patterns, colored pieces, neon, markings, lighting, reflections, "
        "shadows, perspective, borders, or scene composition. Uniform diffuse "
        "color and perfectly repeatable on all four edges."
    )


def ground_prompt(style: str) -> str:
    return (
        "A seamless, tileable PBR BASE-COLOR / ALBEDO texture of generic urban "
        "ground, viewed as an exact flat top-down orthographic material scan. "
        f"Use {GROUND_COLOR_HINTS[style]}. Add only subtle dense material grain "
        "and tiny nondirectional wear consistent with "
        f"{STYLE_DESCRIPTIONS[style]}. The entire square frame must be one "
        "continuous ground material with uniform diffuse color and matching "
        "edges. No scene composition, roads, curbs, grass, objects, perspective, "
        "lighting gradient, reflections, shadows, border, text or large pattern."
    )


def vegetation_prompt(style: str) -> str:
    return (
        "A seamless, tileable PBR BASE-COLOR / ALBEDO texture of dense low grass "
        "and fine urban groundcover, viewed as an exact flat top-down orthographic "
        f"material scan. Use {VEGETATION_COLOR_HINTS[style]}. Preserve a uniform "
        "fine-scale surface with subtle nondirectional variation consistent with "
        f"{STYLE_DESCRIPTIONS[style]}. The entire square frame must be continuous "
        "vegetation with matching edges. No trees, trunks, bushes, flowers, soil "
        "patches, roads, objects, scenery, perspective, shadows, border or text."
    )


def water_prompt(style: str) -> str:
    return (
        "A seamless, tileable PBR BASE-COLOR / ALBEDO texture of stylized anime "
        "water, viewed as an exact flat top-down orthographic material scan. "
        f"Use only {WATER_COLOR_HINTS[style]} as the water color palette. "
        "The entire square image must be uninterrupted water surface from edge "
        "to edge, with soft painterly color variation and gentle connected "
        "ripple patterns. Uniform diffuse lighting, semi-gloss appearance and "
        "matching edges on all four sides. No scenery, no shore, no ground, no "
        "objects, no perspective, no large waves, no reflections, no lighting "
        "gradient, no border, no text."
    )


def skybox_prompt(style: str) -> str:
    return (
        "A complete seamless equirectangular 360-degree skybox texture in exact "
        "2:1 panorama format. Fill the panorama with a coherent atmospheric sky "
        f"whose clouds, colors and subtle details follow {STYLE_DESCRIPTIONS[style]}. "
        "The left and right edges must match perfectly. Sky only, with a clean "
        "low horizon: no ground, terrain, architecture, people, text, frame, "
        "cube faces or visible seam."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reusable per-style tree/lamp images and GLBs plus simple "
            "seamless road base-color textures."
        )
    )
    parser.add_argument(
        "--style",
        choices=(*STYLE_DESCRIPTIONS, "all"),
        default="cyberpunk",
    )
    parser.add_argument("--case-name", default="SE-Stockholm_64")
    parser.add_argument("--tree-variants", type=int, default=2)
    parser.add_argument("--lamp-variants", type=int, default=2)
    parser.add_argument(
        "--asset-kind",
        choices=(
            "all",
            "trees",
            "lamps",
            "surfaces",
            "ground",
            "road",
            "vegetation",
            "water",
            "skybox",
        ),
        default="all",
        help=(
            "Limit generation to one environment asset kind. 'surfaces' selects "
            "ground, road, vegetation and water through the texture LoRA; "
            "'skybox' uses the skybox LoRA."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("plan", "qwen", "hunyuan", "all"),
        default="all",
    )
    parser.add_argument("--qwen-model-path", type=Path, default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--background-model-path",
        type=Path,
        default=DEFAULT_BACKGROUND_MODEL,
    )
    parser.add_argument(
        "--texture-lora-path", type=Path, default=DEFAULT_TEXTURE_LORA
    )
    parser.add_argument(
        "--skybox-lora-path", type=Path, default=DEFAULT_SKYBOX_LORA
    )
    parser.add_argument("--qwen-gpu-groups", default="0,1")
    parser.add_argument("--hunyuan-gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument(
        "--scene-plan",
        type=Path,
        default=None,
        help="Validated Scene Plan JSON whose prompts refine the selected style.",
    )
    parser.add_argument(
        "--enable-vlm-review",
        action="store_true",
        help="Review generated tree and lamp images through an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--vlm-max-attempts",
        type=int,
        default=3,
        help="Maximum generation attempts for each reviewed tree or lamp image.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--material-width", type=int, default=1024)
    parser.add_argument("--material-height", type=int, default=1024)
    parser.add_argument("--skybox-width", type=int, default=2048)
    parser.add_argument("--skybox-height", type=int, default=1024)
    parser.add_argument("--shape-steps", type=int, default=50)
    parser.add_argument("--shape-guidance", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--paint-views", type=int, default=6)
    parser.add_argument("--paint-resolution", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("qwen",), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--job-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-group", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def selected_styles(style: str) -> list[str]:
    return list(STYLE_DESCRIPTIONS) if style == "all" else [style]


def planned_asset_prompt(args: argparse.Namespace, key: str, requirement: str) -> str:
    if args.scene_plan_data is None:
        return requirement
    description = args.scene_plan_data[key]
    return (
        f"Scene Design description:\n{description}\n\n"
        f"Mandatory generation requirements:\n{requirement}"
    )


def environment_root(case_name: str, style: str) -> Path:
    return (
        PROJECT_ROOT
        / "output"
        / case_name
        / f"assets_{style}"
        / "environment"
    )


def build_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    global_asset_index = 0
    for style_index, style in enumerate(selected_styles(args.style)):
        root = environment_root(args.case_name, style)
        if args.asset_kind in ("all", "trees"):
            for variant in range(args.tree_variants):
                output_dir = root / "trees" / f"tree_{variant}"
                jobs.append(
                    {
                        "kind": "tree",
                        "style": style,
                        "variant": variant,
                        "asset_name": f"tree_{variant}",
                        "prompt": planned_asset_prompt(
                            args, "tree", tree_prompt(style, variant)
                        ),
                        "negative_prompt": TREE_NEGATIVE,
                        "seed": args.seed + style_index * 100 + variant,
                        "image_path": output_dir / "image.png",
                        "no_background_path": output_dir / "image_woBG.png",
                        "output_dir": output_dir,
                        "untextured_glb": output_dir / "tree_untextured.glb",
                        "glb_path": output_dir / "tree.glb",
                        "output_stem": "tree",
                        "hunyuan_id": global_asset_index,
                    }
                )
                global_asset_index += 1

        if args.asset_kind in ("all", "lamps"):
            for variant in range(args.lamp_variants):
                output_dir = root / "lamps" / f"lamp_{variant}"
                jobs.append(
                    {
                        "kind": "lamp",
                        "style": style,
                        "variant": variant,
                        "asset_name": f"lamp_{variant}",
                        "prompt": planned_asset_prompt(
                            args, "lamp", lamp_prompt(style, variant)
                        ),
                        "negative_prompt": LAMP_NEGATIVE,
                        "seed": args.seed + style_index * 100 + 20 + variant,
                        "image_path": output_dir / "image.png",
                        "no_background_path": output_dir / "image_woBG.png",
                        "output_dir": output_dir,
                        "untextured_glb": output_dir / "lamp_untextured.glb",
                        "glb_path": output_dir / "lamp.glb",
                        "output_stem": "lamp",
                        "hunyuan_id": global_asset_index,
                    }
                )
                global_asset_index += 1

        if args.asset_kind in ("all", "surfaces", "ground"):
            material_dir = root / "materials"
            jobs.append(
                {
                    "kind": "ground",
                    "style": style,
                    "variant": 0,
                    "asset_name": "ground_basecolor",
                    "prompt": planned_asset_prompt(args, "ground", ground_prompt(style)),
                    "negative_prompt": GROUND_NEGATIVE,
                    "seed": args.seed + style_index * 100 + 40,
                    "image_path": material_dir / "ground_basecolor.png",
                    "record_path": material_dir / "ground_prompt.json",
                    "output_dir": material_dir,
                }
            )
        if args.asset_kind in ("all", "surfaces", "road"):
            road_dir = root / "road"
            jobs.append(
                {
                    "kind": "road",
                    "style": style,
                    "variant": 0,
                    "asset_name": "road_basecolor",
                    "prompt": planned_asset_prompt(args, "road", road_prompt(style)),
                    "negative_prompt": ROAD_NEGATIVE,
                    "seed": args.seed + style_index * 100 + 50,
                    "image_path": road_dir / "road_basecolor.png",
                    "record_path": road_dir / "prompt.json",
                    "output_dir": road_dir,
                }
            )
        if args.asset_kind in ("all", "surfaces", "vegetation"):
            material_dir = root / "materials"
            jobs.append(
                {
                    "kind": "vegetation",
                    "style": style,
                    "variant": 0,
                    "asset_name": "vegetation_basecolor",
                    "prompt": planned_asset_prompt(
                        args, "grass", vegetation_prompt(style)
                    ),
                    "negative_prompt": VEGETATION_NEGATIVE,
                    "seed": args.seed + style_index * 100 + 55,
                    "image_path": material_dir / "vegetation_basecolor.png",
                    "record_path": material_dir / "vegetation_prompt.json",
                    "output_dir": material_dir,
                }
            )
        if args.asset_kind in ("all", "surfaces", "water"):
            material_dir = root / "materials"
            jobs.append(
                {
                    "kind": "water",
                    "style": style,
                    "variant": 0,
                    "asset_name": "water_basecolor",
                    "prompt": planned_asset_prompt(args, "water", water_prompt(style)),
                    "negative_prompt": WATER_NEGATIVE,
                    "seed": args.seed + style_index * 100 + 60,
                    "image_path": material_dir / "water_basecolor.png",
                    "record_path": material_dir / "water_prompt.json",
                    "output_dir": material_dir,
                }
            )
        if args.asset_kind in ("all", "skybox"):
            skybox_dir = root / "skybox"
            jobs.append(
                {
                    "kind": "skybox",
                    "style": style,
                    "variant": 0,
                    "asset_name": "skybox",
                    "prompt": planned_asset_prompt(args, "sky", skybox_prompt(style)),
                    "negative_prompt": SKYBOX_NEGATIVE,
                    "seed": args.seed + style_index * 100 + 70,
                    "image_path": skybox_dir / "skybox.png",
                    "record_path": skybox_dir / "prompt.json",
                    "output_dir": skybox_dir,
                }
            )
    return jobs


def validate_args(args: argparse.Namespace) -> None:
    if args.tree_variants <= 0:
        raise ValueError("--tree-variants must be positive.")
    if args.lamp_variants <= 0:
        raise ValueError("--lamp-variants must be positive.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.material_width <= 0 or args.material_height <= 0:
        raise ValueError("material dimensions must be positive.")
    if args.skybox_width <= 0 or args.skybox_height <= 0:
        raise ValueError("skybox dimensions must be positive.")
    if args.skybox_width != 2 * args.skybox_height:
        raise ValueError("skybox dimensions must use an exact 2:1 ratio.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.true_cfg_scale <= 0:
        raise ValueError("--true-cfg-scale must be positive.")
    if args.vlm_max_attempts <= 0:
        raise ValueError("--vlm-max-attempts must be positive.")
    if args.stage in ("qwen", "all") and not args.dry_run:
        if not args.qwen_model_path.is_dir():
            raise FileNotFoundError(
                f"Qwen Image directory is not ready: {args.qwen_model_path}"
            )
        if (
            args.asset_kind in ("all", "trees", "lamps")
            and not args.background_model_path.is_dir()
        ):
            raise FileNotFoundError(
                "Background-removal model directory is not ready: "
                f"{args.background_model_path}"
            )
        if (
            args.asset_kind in ("all", "surfaces", *SURFACE_KINDS)
            and not args.texture_lora_path.is_file()
        ):
            raise FileNotFoundError(
                f"Texture LoRA file is not ready: {args.texture_lora_path}"
            )
        if args.asset_kind in ("all", "skybox") and not args.skybox_lora_path.is_file():
            raise FileNotFoundError(
                f"Skybox LoRA file is not ready: {args.skybox_lora_path}"
            )


def print_plan(args: argparse.Namespace, jobs: list[dict]) -> None:
    tree_jobs = [job for job in jobs if job["kind"] == "tree"]
    lamp_jobs = [job for job in jobs if job["kind"] == "lamp"]
    ground_jobs = [job for job in jobs if job["kind"] == "ground"]
    road_jobs = [job for job in jobs if job["kind"] == "road"]
    vegetation_jobs = [job for job in jobs if job["kind"] == "vegetation"]
    water_jobs = [job for job in jobs if job["kind"] == "water"]
    skybox_jobs = [job for job in jobs if job["kind"] == "skybox"]
    print("Resolved environment asset pipeline:")
    print(f"  styles:          {', '.join(selected_styles(args.style))}")
    print(f"  tree jobs:       {len(tree_jobs)}")
    print(f"  lamp jobs:       {len(lamp_jobs)}")
    print(f"  ground jobs:     {len(ground_jobs)}")
    print(f"  road jobs:       {len(road_jobs)}")
    print(f"  vegetation jobs: {len(vegetation_jobs)}")
    print(f"  water jobs:      {len(water_jobs)}")
    print(f"  skybox jobs:     {len(skybox_jobs)}")
    print(f"  Qwen model:      {args.qwen_model_path}")
    print(f"  Qwen GPU groups: {args.qwen_gpu_groups}")
    print(f"  Hunyuan GPUs:    {args.hunyuan_gpus}")
    print(f"  Scene Plan:      {args.scene_plan or 'disabled'}")
    print(f"  VLM review:      {args.enable_vlm_review}")
    for job in jobs:
        print(
            f"  {job['style']}/{job['asset_name']}: "
            f"{job['image_path']}"
        )


def write_prompt_record(args: argparse.Namespace, job: dict) -> None:
    job["output_dir"].mkdir(parents=True, exist_ok=True)
    record = {
        "kind": job["kind"],
        "style": job["style"],
        "variant": job["variant"],
        "prompt": job["prompt"],
        "negative_prompt": job["negative_prompt"],
        "seed": job["seed"],
        "model_path": str(args.qwen_model_path),
        "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "width": args.width,
        "height": args.height,
    }
    if "actual_seed" in job:
        record["seed"] = job["actual_seed"]
    if "review_attempts" in job:
        record["vlm_review"] = {
            "enabled": True,
            "kind": job["kind"],
            "attempts": job["review_attempts"],
            "accepted": True,
        }
    with (job["output_dir"] / "prompt.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def parse_qwen_gpu_groups(spec: str) -> list[str]:
    groups = [group.strip() for group in spec.split(";") if group.strip()]
    if not groups:
        raise ValueError("No Qwen GPU group was selected.")
    used = set()
    normalized = []
    for group in groups:
        gpu_ids = parse_gpu_ids(group)
        if len(gpu_ids) != 2:
            raise ValueError(
                f"Each Qwen Image worker needs two GPUs: {group!r}"
            )
        overlap = used.intersection(gpu_ids)
        if overlap:
            raise ValueError(
                f"Qwen GPU groups overlap on: {sorted(overlap)}"
            )
        used.update(gpu_ids)
        normalized.append(",".join(gpu_ids))
    return normalized


def serialize_job(job: dict) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in job.items()
    }


def deserialize_job(job: dict) -> dict:
    result = dict(job)
    for key in (
        "image_path",
        "no_background_path",
        "output_dir",
        "untextured_glb",
        "glb_path",
    ):
        if key in result:
            result[key] = Path(result[key])
    return result


def write_qwen_job_files(
    args: argparse.Namespace,
    groups: list[str],
    jobs: list[dict],
) -> list[tuple[str, Path]]:
    job_root = (
        PROJECT_ROOT
        / "output"
        / args.case_name
        / ".environment_jobs"
    )
    job_root.mkdir(parents=True, exist_ok=True)
    files = []
    for index, group in enumerate(groups):
        shard = jobs[index :: len(groups)]
        if not shard:
            continue
        path = job_root / f"qwen_environment_worker_{index}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {"jobs": [serialize_job(job) for job in shard]},
                file,
                ensure_ascii=False,
                indent=2,
            )
        files.append((group, path))
    return files


def load_worker_jobs(job_file: Path) -> list[dict]:
    with job_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    raw_jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError(f"Expected a non-empty jobs list: {job_file}")
    return [deserialize_job(job) for job in raw_jobs]


def qwen_worker(args: argparse.Namespace, jobs: list[dict]) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from PIL import Image
    from utils.qwen_image import Qwen_Image
    from utils.rembg import BackgroundRemover

    reviewer = (
        OpenAICompatibleVLM.from_environment()
        if args.enable_vlm_review
        else None
    )
    generation_jobs = []
    background_jobs = []
    for job in jobs:
        image_exists = job["image_path"].is_file()
        if args.overwrite or not image_exists:
            generation_jobs.append(job)
        if job["kind"] in ("tree", "lamp") and (
            args.overwrite or not job["no_background_path"].is_file()
        ):
            background_jobs.append(job)

    if reviewer is not None and background_jobs:
        for job in background_jobs:
            if job in generation_jobs:
                continue
            accepted = reviewer.image_review(
                job["image_path"], review_kind=job["kind"]
            )
            print(
                f"[VLM review] existing {job['style']}/{job['asset_name']}: "
                f"{accepted}",
                flush=True,
            )
            if not accepted:
                generation_jobs.append(job)

    if generation_jobs:
        print(
            f"[qwen] Loading local Qwen Image for "
            f"{len(generation_jobs)} job(s) on GPUs {args.gpu_group}",
            flush=True,
        )
        qwen = Qwen_Image(
            model_name=str(args.qwen_model_path),
            device="cuda",
            local_files_only=True,
            device_map="balanced",
        )
        for index, job in enumerate(generation_jobs, start=1):
            job["output_dir"].mkdir(parents=True, exist_ok=True)
            print(
                f"[qwen {index}/{len(generation_jobs)}] "
                f"{job['style']}/{job['asset_name']}",
                flush=True,
            )
            max_attempts = (
                args.vlm_max_attempts
                if reviewer is not None and job["kind"] in ("tree", "lamp")
                else 1
            )
            for attempt in range(max_attempts):
                actual_seed = job["seed"] + attempt
                image = qwen.generate(
                    prompt=job["prompt"],
                    negative_prompt=job["negative_prompt"],
                    seed=actual_seed,
                    image=None,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    memory_optimize=False,
                )
                image.save(job["image_path"])
                job["actual_seed"] = actual_seed
                job["review_attempts"] = attempt + 1
                if reviewer is None or job["kind"] not in ("tree", "lamp"):
                    break
                accepted = reviewer.image_review(
                    job["image_path"], review_kind=job["kind"]
                )
                print(
                    f"[VLM review] {job['style']}/{job['asset_name']} "
                    f"attempt {attempt + 1}/{max_attempts}: {accepted}",
                    flush=True,
                )
                if accepted:
                    break
            else:
                raise RuntimeError(
                    f"VLM review rejected {job['style']}/{job['asset_name']} "
                    f"after {max_attempts} attempt(s)."
                )
            write_prompt_record(args, job)
            print(f"[qwen] Saved: {job['image_path']}", flush=True)
        qwen.unload()
        del qwen

    if background_jobs:
        print(
            f"[rembg] Loading local remover for "
            f"{len(background_jobs)} 3D asset image(s)",
            flush=True,
        )
        remover = BackgroundRemover(
            model_path=str(args.background_model_path),
            device="cuda",
            local_files_only=True,
        )
        for job in background_jobs:
            image = Image.open(job["image_path"]).convert("RGB")
            remover(image).save(job["no_background_path"])
            print(
                f"[rembg] Saved: {job['no_background_path']}",
                flush=True,
            )


def qwen_worker_command(
    args: argparse.Namespace,
    group: str,
    job_file: Path,
) -> list[str]:
    command = [
        str(QWEN_PYTHON),
        str(Path(__file__).resolve()),
        "--worker",
        "qwen",
        "--job-file",
        str(job_file),
        "--gpu-group",
        group,
        "--qwen-model-path",
        str(args.qwen_model_path),
        "--background-model-path",
        str(args.background_model_path),
        "--steps",
        str(args.steps),
        "--true-cfg-scale",
        str(args.true_cfg_scale),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.enable_vlm_review:
        command.extend(
            ["--enable-vlm-review", "--vlm-max-attempts", str(args.vlm_max_attempts)]
        )
    return command


def run_base_qwen_stage(args: argparse.Namespace, jobs: list[dict]) -> None:
    jobs = [job for job in jobs if job["kind"] in ("tree", "lamp")]
    if not jobs:
        print("[qwen] No base Qwen environment jobs were selected; skipping.")
        return
    groups = parse_qwen_gpu_groups(args.qwen_gpu_groups)
    pending = []
    for job in jobs:
        target = (
            job["no_background_path"]
            if job["kind"] in ("tree", "lamp")
            else job["image_path"]
        )
        if args.overwrite or not target.is_file():
            pending.append(job)
    if not pending:
        print("[qwen] All environment images already exist; skipping.")
        return
    if args.dry_run:
        print(f"[qwen dry-run] Pending jobs: {len(pending)}")
        for index, group in enumerate(groups):
            shard = pending[index :: len(groups)]
            print(f"  GPUs {group}: {[job['asset_name'] for job in shard]}")
        return

    job_files = write_qwen_job_files(args, groups, pending)
    commands = []
    for group, job_file in job_files:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = group
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["DIFFUSERS_OFFLINE"] = "1"
        env.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        commands.append(
            (
                f"GPUs {group}",
                qwen_worker_command(args, group, job_file),
                env,
            )
        )
    run_parallel(commands, "Qwen environment")


def lora_job_command(args: argparse.Namespace, job: dict) -> list[str]:
    if job["kind"] in SURFACE_KINDS:
        script = TEXTURE_SCRIPT
        lora_path = args.texture_lora_path
        width = args.material_width
        height = args.material_height
    elif job["kind"] == "skybox":
        script = SKYBOX_SCRIPT
        lora_path = args.skybox_lora_path
        width = args.skybox_width
        height = args.skybox_height
    else:
        raise ValueError(f"Unsupported LoRA environment job: {job['kind']}")
    command = [
        str(QWEN_PYTHON),
        str(script),
        "--model-path",
        str(args.qwen_model_path),
        "--lora-path",
        str(lora_path),
        "--output-path",
        str(job["image_path"]),
        "--record-path",
        str(job["record_path"]),
        "--prompt",
        job["prompt"],
        "--negative-prompt",
        job["negative_prompt"],
        "--seed",
        str(job["seed"]),
        "--width",
        str(width),
        "--height",
        str(height),
        "--steps",
        str(args.steps),
        "--true-cfg-scale",
        str(args.true_cfg_scale),
    ]
    if job["kind"] in SURFACE_KINDS:
        command.extend(["--surface-kind", job["kind"]])
    return command


def run_lora_material_stage(args: argparse.Namespace, jobs: list[dict]) -> None:
    jobs = [
        job for job in jobs if job["kind"] in (*SURFACE_KINDS, "skybox")
    ]
    pending = [
        job for job in jobs if args.overwrite or not job["image_path"].is_file()
    ]
    if not pending:
        print("[lora] All surface and skybox textures already exist; skipping.")
        return
    groups = parse_qwen_gpu_groups(args.qwen_gpu_groups)
    if args.dry_run:
        print(f"[lora dry-run] Pending jobs: {len(pending)}")
        for index, job in enumerate(pending):
            print(f"  GPUs {groups[index % len(groups)]}: {job['style']}/{job['asset_name']}")
        return

    grouped: dict[str, list[dict]] = {group: [] for group in groups}
    for index, job in enumerate(pending):
        grouped[groups[index % len(groups)]].append(job)
    wave_count = max(len(shard) for shard in grouped.values())
    for wave in range(wave_count):
        commands = []
        for group, shard in grouped.items():
            if wave >= len(shard):
                continue
            job = shard[wave]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = group
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["DIFFUSERS_OFFLINE"] = "1"
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            commands.append(
                (
                    f"{job['style']}/{job['asset_name']} GPUs {group}",
                    lora_job_command(args, job),
                    env,
                )
            )
        run_parallel(commands, "Qwen LoRA materials")


def run_qwen_stage(args: argparse.Namespace, jobs: list[dict]) -> None:
    run_base_qwen_stage(args, jobs)
    run_lora_material_stage(args, jobs)


def parse_gpu_ids(spec: str) -> list[str]:
    gpu_ids = [item.strip() for item in spec.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU list must contain distinct comma-separated IDs.")
    invalid = [gpu_id for gpu_id in gpu_ids if not gpu_id.isdecimal()]
    if invalid:
        raise ValueError(f"Only non-negative physical GPU IDs are supported: {invalid}")
    return gpu_ids


def write_hunyuan_job_files(
    args: argparse.Namespace,
    asset_jobs: list[dict],
) -> list[tuple[str, Path]]:
    gpu_ids = parse_gpu_ids(args.hunyuan_gpus)
    job_root = (
        PROJECT_ROOT
        / "output"
        / args.case_name
        / ".environment_jobs"
    )
    job_root.mkdir(parents=True, exist_ok=True)
    files = []
    for index, gpu_id in enumerate(gpu_ids):
        shard = asset_jobs[index :: len(gpu_ids)]
        if not shard:
            continue
        path = job_root / f"hunyuan_environment_gpu_{gpu_id}.json"
        payload = {
            "jobs": [
                {
                    "building_id": job["hunyuan_id"],
                    "style": job["style"],
                    "asset_name": job["asset_name"],
                    "output_stem": job["output_stem"],
                    "input_image": str(job["no_background_path"]),
                    "output_dir": str(job["output_dir"]),
                }
                for job in shard
            ]
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        files.append((gpu_id, path))
    return files


def hunyuan_command(
    args: argparse.Namespace,
    gpu_id: str,
    job_file: Path,
    stage: str,
) -> list[str]:
    command = [
        str(HUNYUAN_PYTHON),
        str(HUNYUAN_SCRIPT),
        "--batch-job-file",
        str(job_file),
        "--gpu",
        gpu_id,
        "--stage",
        stage,
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
    if args.skip_existing:
        command.append("--skip-existing")
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_parallel(commands, label: str) -> None:
    processes = []
    try:
        for item in commands:
            worker, command = item[:2]
            env = item[2] if len(item) > 2 else None
            print(f"[{label}/{worker}] {shlex.join(command)}", flush=True)
            processes.append(
                (
                    worker,
                    command,
                    subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        start_new_session=True,
                    ),
                )
            )
        pending = set(range(len(processes)))
        while pending:
            for index in list(pending):
                worker, command, process = processes[index]
                code = process.poll()
                if code is None:
                    continue
                pending.remove(index)
                if code != 0:
                    raise subprocess.CalledProcessError(code, command)
                print(f"[{label}/{worker}] completed", flush=True)
            if pending:
                time.sleep(0.5)
    except BaseException:
        for _worker, _command, process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _worker, _command, process in processes:
            if process.poll() is None:
                process.wait()
        raise


def run_hunyuan_stage(args: argparse.Namespace, jobs: list[dict]) -> None:
    asset_jobs = [
        job for job in jobs if job["kind"] in ("tree", "lamp")
    ]
    if not asset_jobs:
        print("[hunyuan] No tree or lamp jobs were selected; skipping.")
        return
    missing = [
        job["no_background_path"]
        for job in asset_jobs
        if not job["no_background_path"].is_file()
    ]
    if missing and not args.dry_run:
        raise FileNotFoundError(
            "Tree/lamp images must be generated before Hunyuan3D; first missing: "
            f"{missing[0]}"
        )
    job_files = write_hunyuan_job_files(args, asset_jobs)
    for stage in ("shape", "paint"):
        commands = [
            (
                f"GPU {gpu_id}",
                hunyuan_command(args, gpu_id, job_file, stage),
            )
            for gpu_id, job_file in job_files
        ]
        run_parallel(commands, f"Hunyuan environment {stage}")


def main() -> None:
    args = parse_args()
    args.qwen_model_path = args.qwen_model_path.expanduser().resolve()
    args.background_model_path = (
        args.background_model_path.expanduser().resolve()
    )
    args.texture_lora_path = args.texture_lora_path.expanduser().resolve()
    args.skybox_lora_path = args.skybox_lora_path.expanduser().resolve()
    if args.scene_plan is not None:
        args.scene_plan = args.scene_plan.expanduser().resolve()
        if args.scene_plan.is_file():
            args.scene_plan_data = load_scene_plan(args.scene_plan).to_dict()
        elif args.dry_run:
            args.scene_plan_data = {
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
        else:
            raise FileNotFoundError(f"Scene Plan not found: {args.scene_plan}")
    else:
        args.scene_plan_data = None
    validate_args(args)
    if args.worker == "qwen":
        if not args.job_file or not args.gpu_group:
            raise ValueError("Qwen worker requires --job-file and --gpu-group.")
        qwen_worker(
            args,
            load_worker_jobs(args.job_file.expanduser().resolve()),
        )
        return

    jobs = build_jobs(args)
    print_plan(args, jobs)

    if args.stage == "plan":
        print("Planning completed; no files or models were loaded.")
        return
    if args.stage in ("qwen", "all"):
        run_qwen_stage(args, jobs)
    if args.stage in ("hunyuan", "all"):
        run_hunyuan_stage(args, jobs)


if __name__ == "__main__":
    main()
