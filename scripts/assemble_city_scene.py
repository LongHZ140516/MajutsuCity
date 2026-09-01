#!/usr/bin/env python3
"""Build a semantic Blender city scene from a layout and generated GLBs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from _runtime import env_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNCITY_PYTHON = env_path(
    "MAJUTSUCITY_RENDER_PYTHON", "envs/majutsucity/bin/python"
)
BLENDER_PYTHON = env_path(
    "MAJUTSUCITY_BLENDER_PYTHON", "envs/majutsucity/bin/python"
)
DEFAULT_LAYOUT = PROJECT_ROOT / "examples/pair_cases/layout/SE-Stockholm_64.png"
DEFAULT_DEPTH = PROJECT_ROOT / "examples/pair_cases/depth/SE-Stockholm_64.png"

STYLE_PALETTES = {
    "christmas": {
        "terrain": (0.32, 0.30, 0.28, 1.0),
        "ground": (0.72, 0.77, 0.79, 1.0),
        "road": (0.18, 0.20, 0.22, 1.0),
        "vegetation": (0.08, 0.28, 0.14, 1.0),
        "water": (0.18, 0.45, 0.65, 0.75),
        "foliage": (0.05, 0.35, 0.14, 1.0),
        "lamp": (1.0, 0.58, 0.16, 1.0),
        "sky": (0.20, 0.30, 0.48, 1.0),
    },
    "cyberpunk": {
        "terrain": (0.035, 0.045, 0.065, 1.0),
        "ground": (0.07, 0.09, 0.12, 1.0),
        "road": (0.025, 0.035, 0.055, 1.0),
        "vegetation": (0.02, 0.18, 0.17, 1.0),
        "water": (0.02, 0.32, 0.48, 0.72),
        "foliage": (0.02, 0.42, 0.34, 1.0),
        "lamp": (0.05, 0.95, 1.0, 1.0),
        "sky": (0.012, 0.018, 0.055, 1.0),
    },
    "european": {
        "terrain": (0.25, 0.22, 0.18, 1.0),
        "ground": (0.46, 0.40, 0.32, 1.0),
        "road": (0.16, 0.17, 0.18, 1.0),
        "vegetation": (0.13, 0.33, 0.10, 1.0),
        "water": (0.14, 0.34, 0.53, 0.72),
        "foliage": (0.18, 0.42, 0.11, 1.0),
        "lamp": (1.0, 0.64, 0.28, 1.0),
        "sky": (0.30, 0.48, 0.68, 1.0),
    },
    "genshin": {
        "terrain": (0.28, 0.23, 0.20, 1.0),
        "ground": (0.55, 0.45, 0.30, 1.0),
        "road": (0.20, 0.18, 0.25, 1.0),
        "vegetation": (0.16, 0.48, 0.22, 1.0),
        "water": (0.08, 0.52, 0.76, 0.72),
        "foliage": (0.18, 0.62, 0.30, 1.0),
        "lamp": (0.30, 0.82, 1.0, 1.0),
        "sky": (0.32, 0.50, 0.82, 1.0),
    },
    "ghibli": {
        "terrain": (0.34, 0.28, 0.20, 1.0),
        "ground": (0.58, 0.49, 0.34, 1.0),
        "road": (0.25, 0.24, 0.22, 1.0),
        "vegetation": (0.20, 0.50, 0.16, 1.0),
        "water": (0.14, 0.55, 0.70, 0.72),
        "foliage": (0.24, 0.60, 0.18, 1.0),
        "lamp": (1.0, 0.68, 0.30, 1.0),
        "sky": (0.42, 0.60, 0.78, 1.0),
    },
    "halloween": {
        "terrain": (0.10, 0.07, 0.08, 1.0),
        "ground": (0.20, 0.12, 0.08, 1.0),
        "road": (0.045, 0.035, 0.05, 1.0),
        "vegetation": (0.12, 0.18, 0.04, 1.0),
        "water": (0.12, 0.18, 0.26, 0.76),
        "foliage": (0.20, 0.25, 0.03, 1.0),
        "lamp": (1.0, 0.24, 0.03, 1.0),
        "sky": (0.07, 0.025, 0.10, 1.0),
    },
    "mc": {
        "terrain": (0.28, 0.22, 0.14, 1.0),
        "ground": (0.46, 0.34, 0.20, 1.0),
        "road": (0.18, 0.18, 0.18, 1.0),
        "vegetation": (0.16, 0.48, 0.12, 1.0),
        "water": (0.05, 0.35, 0.80, 0.78),
        "foliage": (0.12, 0.52, 0.10, 1.0),
        "lamp": (1.0, 0.75, 0.22, 1.0),
        "sky": (0.32, 0.58, 0.82, 1.0),
    },
    "modern": {
        "terrain": (0.22, 0.24, 0.23, 1.0),
        "ground": (0.46, 0.48, 0.46, 1.0),
        "road": (0.14, 0.16, 0.17, 1.0),
        "vegetation": (0.11, 0.36, 0.16, 1.0),
        "water": (0.08, 0.38, 0.58, 0.72),
        "foliage": (0.12, 0.43, 0.18, 1.0),
        "lamp": (0.82, 0.92, 1.0, 1.0),
        "sky": (0.48, 0.64, 0.76, 1.0),
    },
    "springfest": {
        "terrain": (0.30, 0.19, 0.13, 1.0),
        "ground": (0.50, 0.30, 0.17, 1.0),
        "road": (0.16, 0.09, 0.08, 1.0),
        "vegetation": (0.17, 0.43, 0.12, 1.0),
        "water": (0.08, 0.36, 0.58, 0.72),
        "foliage": (0.20, 0.50, 0.12, 1.0),
        "lamp": (1.0, 0.12, 0.03, 1.0),
        "sky": (0.58, 0.18, 0.15, 1.0),
    },
    "technical": {
        "terrain": (0.22, 0.25, 0.28, 1.0),
        "ground": (0.58, 0.64, 0.68, 1.0),
        "road": (0.12, 0.16, 0.20, 1.0),
        "vegetation": (0.10, 0.48, 0.48, 1.0),
        "water": (0.02, 0.55, 0.80, 0.70),
        "foliage": (0.08, 0.65, 0.60, 1.0),
        "lamp": (0.16, 0.80, 1.0, 1.0),
        "sky": (0.45, 0.65, 0.78, 1.0),
    },
    "vangogh": {
        "terrain": (0.24, 0.16, 0.18, 1.0),
        "ground": (0.58, 0.38, 0.16, 1.0),
        "road": (0.08, 0.12, 0.30, 1.0),
        "vegetation": (0.18, 0.42, 0.16, 1.0),
        "water": (0.02, 0.24, 0.68, 0.75),
        "foliage": (0.20, 0.48, 0.15, 1.0),
        "lamp": (1.0, 0.64, 0.06, 1.0),
        "sky": (0.04, 0.12, 0.42, 1.0),
    },
}


def available_styles(case_name: str) -> list[str]:
    case_dir = PROJECT_ROOT / "output" / case_name
    return sorted(
        path.name.removeprefix("assets_")
        for path in case_dir.glob("assets_*")
        if path.is_dir() and (path / "buildings").is_dir()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract building placements from a layout/depth pair, import one "
            "style's GLBs in Blender, and save a scene plus preview."
        )
    )
    parser.add_argument("--style", default="mc", help="Style name or 'all'.")
    parser.add_argument("--case-name", default="SE-Stockholm_64")
    parser.add_argument("--assets-name", default=None)
    parser.add_argument("--layout-path", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--depth-path", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument(
        "--layout-palette",
        choices=("citydreamer", "majutsu", "auto"),
        default="citydreamer",
    )
    parser.add_argument(
        "--footprint-fill",
        type=float,
        default=0.9,
        help="Fraction of each oriented layout box occupied by its building.",
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to layout depth-derived building heights.",
    )
    parser.add_argument(
        "--building-scale-mode",
        choices=("uniform", "depth"),
        default="depth",
        help=(
            "depth independently matches Z to the depth-map height (default); "
            "uniform preserves the source Hunyuan asset proportions."
        ),
    )
    parser.add_argument(
        "--surface-cell-size",
        type=int,
        default=2,
        help="Layout pixel aggregation used for semantic surface meshes (default: 2).",
    )
    parser.add_argument(
        "--no-environment",
        action="store_true",
        help="Disable semantic terrain/road/vegetation/water generation.",
    )
    parser.add_argument(
        "--no-props",
        action="store_true",
        help="Disable generated vegetation props, road trees and street lights.",
    )
    parser.add_argument(
        "--allow-missing-buildings",
        action="store_true",
        help=(
            "Skip layout placements whose building GLB is missing. Disabled by "
            "default so production scene assembly remains strict."
        ),
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker",
        choices=("prepare", "blender"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--placements-file", type=Path, default=None)
    return parser.parse_args()


def placement_file_for_case(case_name: str) -> Path:
    return (
        PROJECT_ROOT
        / "output"
        / case_name
        / "scenes"
        / "layout_placements.json"
    )


def prepare_placements(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(PROJECT_ROOT))
    from utils.buildings_extract import extract_objects_info
    from utils.function import image_to_mask

    layout = Image.open(args.layout_path).convert("RGB")
    depth = np.asarray(Image.open(args.depth_path).convert("L"), dtype=np.float32)
    layout_mask = image_to_mask(layout, palette_name=args.layout_palette)
    _, _, objects = extract_objects_info("building", layout_mask, depth)
    placements = [
        {
            "building_id": int(obj.id),
            "cx": float(obj.cx),
            "cy": float(obj.cy),
            "footprint_w": float(obj.w),
            "footprint_h": float(obj.h),
            "angle_deg_image": float(obj.angle),
            "height": float(obj.height),
        }
        for obj in objects
    ]
    if not placements:
        raise RuntimeError("No buildings were extracted from the layout.")

    output = {
        "case_name": args.case_name,
        "layout_path": str(args.layout_path),
        "depth_path": str(args.depth_path),
        "layout_palette": args.layout_palette,
        "layout_width": layout.width,
        "layout_height": layout.height,
        "building_count": len(placements),
        "placements": placements,
    }
    args.placements_file.parent.mkdir(parents=True, exist_ok=True)
    with args.placements_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(
        f"Saved {len(placements)} layout placements: {args.placements_file}",
        flush=True,
    )


def clear_blender_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def import_joined_mesh(bpy, glb_path: Path, object_name: str):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb_path), loglevel=50)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh object imported from {glb_path}")

    for obj in meshes:
        world_matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world_matrix

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    building = bpy.context.view_layer.objects.active
    building.name = object_name
    if building.data:
        building.data.name = f"{object_name}_mesh"

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for obj in imported:
        if obj != building and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    xs = [vertex.co.x for vertex in building.data.vertices]
    ys = [vertex.co.y for vertex in building.data.vertices]
    zs = [vertex.co.z for vertex in building.data.vertices]
    if not xs:
        raise RuntimeError(f"Imported mesh has no vertices: {glb_path}")
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    min_z = min(zs)
    for vertex in building.data.vertices:
        vertex.co.x -= center_x
        vertex.co.y -= center_y
        vertex.co.z -= min_z

    dimensions = (
        max(max(xs) - min(xs), 1e-6),
        max(max(ys) - min(ys), 1e-6),
        max(max(zs) - min(zs), 1e-6),
    )
    return building, dimensions


def create_material(bpy, name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.85
    return material


def terrain_height(x: float, y: float) -> float:
    """Flat city datum shared by terrain, surfaces, props and buildings."""
    del x, y
    return 0.0


def ensure_collection(bpy, name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def link_only_to_collection(obj, collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def create_tiled_image_material(
    bpy,
    name: str,
    image_path: Path,
    *,
    tiles_x: float,
    tiles_y: float,
    roughness: float = 0.72,
    transmission: float = 0.0,
    bump_strength: float = 0.18,
    bump_distance: float = 0.08,
):
    """Create a repeatable surface material from a generated base-color map."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    image_node = nodes.new("ShaderNodeTexImage")
    rgb_to_bw = nodes.new("ShaderNodeRGBToBW")
    bump = nodes.new("ShaderNodeBump")

    image_node.image = bpy.data.images.load(
        str(image_path), check_existing=True
    )
    image_node.extension = "REPEAT"
    mapping.inputs["Scale"].default_value = (
        max(tiles_x, 1.0),
        max(tiles_y, 1.0),
        1.0,
    )
    bsdf.inputs["Roughness"].default_value = roughness
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = transmission
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = bump_distance

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], image_node.inputs["Vector"])
    links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(image_node.outputs["Color"], rgb_to_bw.inputs["Color"])
    links.new(rgb_to_bw.outputs["Val"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    material["source_image"] = str(image_path)
    return material


def create_mesh_object(
    bpy,
    collection,
    name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
    material,
):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if material is not None:
        mesh.materials.append(material)
    return obj


def create_base_terrain(bpy, width: int, height: int, material):
    collection = ensure_collection(bpy, "terrain")
    step = 8
    xs = list(range(0, width, step))
    ys = list(range(0, height, step))
    if xs[-1] != width:
        xs.append(width)
    if ys[-1] != height:
        ys.append(height)
    vertices = [
        (float(x), -float(y), terrain_height(float(x), -float(y)) - 0.22)
        for y in ys
        for x in xs
    ]
    row_size = len(xs)
    faces = []
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            lower_left = row * row_size + column
            faces.append(
                (
                    lower_left,
                    lower_left + row_size,
                    lower_left + row_size + 1,
                    lower_left + 1,
                )
            )
    return create_mesh_object(
        bpy, collection, "terrain_flat", vertices, faces, material
    )


def create_semantic_surface(
    bpy,
    label: str,
    mask,
    cell_size: int,
    z_offset: float,
    material,
) -> int:
    """Raster-to-mesh conversion preserving the layout without Shapely."""
    collection = ensure_collection(bpy, label)
    height, width = mask.shape
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    vertex_indices: dict[tuple[int, int], int] = {}

    def vertex_index(x: int, y: int) -> int:
        key = (x, y)
        if key not in vertex_indices:
            vertex_indices[key] = len(vertices)
            vertices.append(
                (
                    float(x),
                    -float(y),
                    terrain_height(float(x), -float(y)) + z_offset,
                )
            )
        return vertex_indices[key]

    cell_count = 0
    for y0 in range(0, height, cell_size):
        y1 = min(y0 + cell_size, height)
        for x0 in range(0, width, cell_size):
            x1 = min(x0 + cell_size, width)
            block = mask[y0:y1, x0:x1]
            # A quarter-cell threshold retains narrow roads while avoiding most
            # single-pixel color noise at label boundaries.
            if not block.any() or float(block.mean()) < 0.25:
                continue
            faces.append(
                (
                    vertex_index(x0, y0),
                    vertex_index(x0, y1),
                    vertex_index(x1, y1),
                    vertex_index(x1, y0),
                )
            )
            cell_count += 1

    if faces:
        create_mesh_object(
            bpy,
            collection,
            f"{label}_layout_surface",
            vertices,
            faces,
            material,
        )
    return cell_count


def setup_world_sky(bpy, skybox_path: Path) -> str:
    world = bpy.context.scene.world or bpy.data.worlds.new("style_sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    texture_coordinates = nodes.new("ShaderNodeTexCoord")
    environment_texture = nodes.new("ShaderNodeTexEnvironment")
    environment_texture.image = bpy.data.images.load(
        str(skybox_path), check_existing=True
    )
    environment_texture.projection = "EQUIRECTANGULAR"
    links.new(
        texture_coordinates.outputs["Generated"],
        environment_texture.inputs["Vector"],
    )
    links.new(
        environment_texture.outputs["Color"],
        background.inputs["Color"],
    )
    source = str(skybox_path)
    world["source_image"] = source
    background.inputs["Strength"].default_value = 0.55
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return source


def sample_mask_points(
    mask,
    *,
    spacing: float,
    limit: int,
    seed: int,
    boundary: bool = False,
) -> list[tuple[float, float]]:
    import cv2
    import numpy as np

    selected_mask = mask.astype(np.uint8)
    if boundary:
        eroded = cv2.erode(
            selected_mask,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        selected_mask = selected_mask & (1 - eroded)
    coordinates = np.argwhere(selected_mask > 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(coordinates)

    selected: list[tuple[float, float]] = []
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    grid_size = max(spacing, 1.0)
    for row, column in coordinates:
        point = (float(column) + 0.5, -float(row) - 0.5)
        cell = (
            int(math.floor(point[0] / grid_size)),
            int(math.floor(point[1] / grid_size)),
        )
        too_close = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((cell[0] + dx, cell[1] + dy), []):
                    if (
                        (point[0] - other[0]) ** 2
                        + (point[1] - other[1]) ** 2
                        < spacing**2
                    ):
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            continue
        selected.append(point)
        grid.setdefault(cell, []).append(point)
        if len(selected) >= limit:
            break
    return selected


def load_generated_asset_templates(
    bpy,
    environment_dir: Path,
    *,
    category: str,
    asset_stem: str,
):
    tree_paths = sorted(
        (environment_dir / category).glob(
            f"{asset_stem}_*/{asset_stem}.glb"
        )
    )
    if not tree_paths:
        return []

    templates_collection = ensure_collection(bpy, "_templates")
    templates_collection.hide_render = False
    templates_collection.hide_viewport = False
    templates = []
    for index, tree_path in enumerate(tree_paths):
        tree, dimensions = import_joined_mesh(
            bpy,
            tree_path,
            f"generated_{asset_stem}_{index}_template",
        )
        link_only_to_collection(tree, templates_collection)
        templates.append(
            {
                "object": tree,
                "dimensions": dimensions,
                "source_glb": str(tree_path),
            }
        )
    templates_collection.hide_render = True
    templates_collection.hide_viewport = True
    return templates


def scatter_generated_assets(
    bpy,
    templates: list[dict],
    collection_name: str,
    points: list[tuple[float, float]],
    *,
    target_height: float,
    seed: int,
) -> int:
    import random

    collection = ensure_collection(bpy, collection_name)
    rng = random.Random(seed)
    for index, (x, y) in enumerate(points):
        template = templates[index % len(templates)]
        template_object = template["object"]
        source_height = max(float(template["dimensions"][2]), 1e-6)
        height = target_height * rng.uniform(0.78, 1.22)
        scale = height / source_height
        obj = bpy.data.objects.new(
            f"{collection_name}_{index:04d}",
            template_object.data,
        )
        collection.objects.link(obj)
        obj.scale = (scale, scale, scale)
        obj.rotation_mode = "XYZ"
        obj.rotation_euler[2] = rng.uniform(0.0, math.tau)
        obj.location = (x, y, terrain_height(x, y) + 0.10)
    return len(points)


def create_environment(bpy, args: argparse.Namespace, style: str):
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(PROJECT_ROOT))
    from utils.function import extract_lmask, image_to_mask

    assets_name = args.assets_name or f"assets_{style}"
    environment_dir = (
        PROJECT_ROOT
        / "output"
        / args.case_name
        / assets_name
        / "environment"
    )
    surface_texture_paths = {
        "ground": environment_dir / "materials" / "ground_basecolor.png",
        "road": environment_dir / "road" / "road_basecolor.png",
        "vegetation": (
            environment_dir / "materials" / "vegetation_basecolor.png"
        ),
        "water": environment_dir / "materials" / "water_basecolor.png",
    }
    skybox_path = environment_dir / "skybox" / "skybox.png"
    missing_textures = [
        path
        for path in (*surface_texture_paths.values(), skybox_path)
        if not path.is_file()
    ]
    if missing_textures:
        raise FileNotFoundError(
            "Generated environment textures are required for scene assembly; "
            f"first missing: {missing_textures[0]}. Run the environment stage "
            "or generate --asset-kind surfaces/skybox before assembling."
        )
    layout = Image.open(args.layout_path).convert("RGB")
    layout_mask = image_to_mask(layout, palette_name=args.layout_palette)
    masks = {
        "ground": extract_lmask("ground", layout_mask),
        "road": extract_lmask("traffic_roads", layout_mask),
        "vegetation": extract_lmask("vegetation", layout_mask),
        "water": extract_lmask("water", layout_mask),
    }
    materials = {
        "terrain": create_tiled_image_material(
            bpy,
            f"{style}_terrain_generated",
            surface_texture_paths["ground"],
            tiles_x=layout.width / 64.0,
            tiles_y=layout.height / 64.0,
            roughness=0.82,
            bump_strength=0.12,
            bump_distance=0.06,
        ),
        "ground": create_tiled_image_material(
            bpy,
            f"{style}_ground_generated",
            surface_texture_paths["ground"],
            tiles_x=layout.width / 64.0,
            tiles_y=layout.height / 64.0,
            roughness=0.80,
            bump_strength=0.12,
            bump_distance=0.06,
        ),
        "road": create_tiled_image_material(
            bpy,
            f"{style}_road_generated",
            surface_texture_paths["road"],
            tiles_x=layout.width / 64.0,
            tiles_y=layout.height / 64.0,
            roughness=0.72,
            bump_strength=0.14,
            bump_distance=0.07,
        ),
        "vegetation": create_tiled_image_material(
            bpy,
            f"{style}_vegetation_generated",
            surface_texture_paths["vegetation"],
            tiles_x=layout.width / 48.0,
            tiles_y=layout.height / 48.0,
            roughness=0.88,
            bump_strength=0.18,
            bump_distance=0.08,
        ),
        "water": create_tiled_image_material(
            bpy,
            f"{style}_water_generated",
            surface_texture_paths["water"],
            tiles_x=layout.width / 96.0,
            tiles_y=layout.height / 96.0,
            roughness=0.16,
            transmission=0.25,
            bump_strength=0.10,
            bump_distance=0.04,
        ),
    }
    create_base_terrain(bpy, layout.width, layout.height, materials["terrain"])
    offsets = {"ground": 0.02, "road": 0.055, "vegetation": 0.04, "water": 0.08}
    surface_cells = {
        label: create_semantic_surface(
            bpy,
            label,
            mask,
            args.surface_cell_size,
            offsets[label],
            materials[label],
        )
        for label, mask in masks.items()
    }
    sky_source = setup_world_sky(bpy, skybox_path)

    prop_counts = {"plant": 0, "aleetree": 0, "light": 0}
    generated_trees = []
    generated_lamps = []
    if not args.no_props:
        generated_trees = load_generated_asset_templates(
            bpy,
            environment_dir,
            category="trees",
            asset_stem="tree",
        )
        generated_lamps = load_generated_asset_templates(
            bpy,
            environment_dir,
            category="lamps",
            asset_stem="lamp",
        )
        if not generated_trees or not generated_lamps:
            missing_kind = "trees" if not generated_trees else "lamps"
            raise FileNotFoundError(
                f"Generated {missing_kind} are required for scene props in "
                f"{environment_dir}. Run the environment stage first or use "
                "--no-props explicitly."
            )
        plant_points = sample_mask_points(
            masks["vegetation"], spacing=9.0, limit=180, seed=1201
        )
        road_edge_points = sample_mask_points(
            masks["road"], spacing=13.0, limit=150, seed=2402, boundary=True
        )
        tree_points = road_edge_points[::2][:55]
        lamp_points = road_edge_points[1::2][:70]
        prop_counts["plant"] = scatter_generated_assets(
            bpy,
            generated_trees,
            "plant",
            plant_points,
            target_height=7.0,
            seed=3403,
        )
        prop_counts["aleetree"] = scatter_generated_assets(
            bpy,
            generated_trees,
            "aleetree",
            tree_points,
            target_height=9.0,
            seed=4504,
        )
        prop_counts["light"] = scatter_generated_assets(
            bpy,
            generated_lamps,
            "light",
            lamp_points,
            target_height=5.5,
            seed=5605,
        )
    return {
        "surface_cells": surface_cells,
        "semantic_pixels": {
            label: int(np.count_nonzero(mask)) for label, mask in masks.items()
        },
        "props": prop_counts,
        "tree_assets": [
            template["source_glb"] for template in generated_trees
        ],
        "tree_source": "generated_glb" if generated_trees else "disabled",
        "lamp_assets": [
            template["source_glb"] for template in generated_lamps
        ],
        "lamp_source": "generated_glb" if generated_lamps else "disabled",
        "terrain_material": str(surface_texture_paths["ground"]),
        "ground_material": str(surface_texture_paths["ground"]),
        "road_material": str(surface_texture_paths["road"]),
        "vegetation_material": str(surface_texture_paths["vegetation"]),
        "water_material": str(surface_texture_paths["water"]),
        "terrain": "flat",
        "sky": sky_source,
    }


def point_camera_at(camera, target, Vector) -> None:
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera_and_lighting(
    bpy,
    width: float,
    height: float,
    max_building_height: float,
) -> None:
    from mathutils import Vector

    center = Vector((width / 2.0, -height / 2.0, max_building_height * 0.2))
    extent = max(width, height)

    camera_data = bpy.data.cameras.new("isometric_camera")
    camera = bpy.data.objects.new("isometric_camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = center + Vector((extent * 0.75, -extent * 0.75, extent * 0.8))
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = extent * 1.5
    point_camera_at(camera, center, Vector)
    bpy.context.scene.camera = camera

    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 2.5
    sun_data.angle = math.radians(20)
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(28), math.radians(-20), math.radians(135))

    area_data = bpy.data.lights.new("soft_fill", type="AREA")
    area_data.energy = 1400
    area_data.shape = "DISK"
    area_data.size = extent
    area = bpy.data.objects.new("soft_fill", area_data)
    bpy.context.scene.collection.objects.link(area)
    area.location = center + Vector((-extent * 0.25, extent * 0.2, extent * 0.9))
    point_camera_at(area, center, Vector)


def configure_render(bpy, preview_path: Path, resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(preview_path)
    scene.render.film_transparent = False
    # These renders are placement previews rather than final beauty shots.
    # Keeping EEVEE samples modest makes batch style validation much faster.
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


SCENE_CACHE_VERSION = 3


def file_fingerprint(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def scene_cache_key(
    args: argparse.Namespace,
    style: str,
    placement_data: dict,
    active_placements: list[dict],
    missing_ids: list[int],
    buildings_dir: Path,
) -> str:
    assets_name = args.assets_name or f"assets_{style}"
    environment_dir = (
        PROJECT_ROOT / "output" / args.case_name / assets_name / "environment"
    )
    source_paths = [args.layout_path, args.depth_path]
    source_paths.extend(
        buildings_dir / f"building_{item['building_id']}" / "building.glb"
        for item in active_placements
    )
    if not args.no_environment:
        source_paths.append(
            environment_dir / "materials" / "ground_basecolor.png"
        )
        source_paths.append(environment_dir / "road" / "road_basecolor.png")
        source_paths.append(
            environment_dir / "materials" / "vegetation_basecolor.png"
        )
        source_paths.append(
            environment_dir / "materials" / "water_basecolor.png"
        )
        source_paths.append(environment_dir / "skybox" / "skybox.png")
        if not args.no_props:
            source_paths.extend(
                sorted((environment_dir / "trees").glob("tree_*/tree.glb"))
            )
            source_paths.extend(
                sorted((environment_dir / "lamps").glob("lamp_*/lamp.glb"))
            )

    payload = {
        "version": SCENE_CACHE_VERSION,
        "style": style,
        "case_name": args.case_name,
        "assets_name": assets_name,
        "layout_palette": args.layout_palette,
        "placement_data": placement_data,
        "active_building_ids": [
            int(item["building_id"]) for item in active_placements
        ],
        "missing_building_ids": missing_ids,
        "assembly_config": {
            "footprint_fill": args.footprint_fill,
            "height_scale": args.height_scale,
            "building_scale_mode": args.building_scale_mode,
            "surface_cell_size": args.surface_cell_size,
            "resolution": args.resolution,
            "no_environment": args.no_environment,
            "no_props": args.no_props,
        },
        "source_files": [file_fingerprint(path) for path in source_paths],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def report_cache_key(report_path: Path) -> str | None:
    if not report_path.is_file():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    value = report.get("scene_cache_key")
    return value if isinstance(value, str) else None


def assemble_blender_scene(args: argparse.Namespace) -> None:
    import bpy

    with args.placements_file.open("r", encoding="utf-8") as file:
        placement_data = json.load(file)
    placements = placement_data["placements"]
    style = args.style
    assets_name = args.assets_name or f"assets_{style}"
    buildings_dir = (
        PROJECT_ROOT / "output" / args.case_name / assets_name / "buildings"
    )
    scene_dir = PROJECT_ROOT / "output" / args.case_name / "scenes" / style
    blend_path = scene_dir / "city_scene.blend"
    preview_path = scene_dir / "preview.png"
    report_path = scene_dir / "assembly_report.json"

    missing_placements = [
        item
        for item in placements
        if not (
            buildings_dir / f"building_{item['building_id']}" / "building.glb"
        ).is_file()
    ]
    missing_paths = [
        buildings_dir / f"building_{item['building_id']}" / "building.glb"
        for item in missing_placements
    ]
    if missing_paths and not args.allow_missing_buildings:
        raise FileNotFoundError(
            f"{style} is missing {len(missing_paths)} building GLBs; "
            f"first: {missing_paths[0]}"
        )
    missing_ids = [int(item["building_id"]) for item in missing_placements]
    active_placements = [
        item for item in placements if int(item["building_id"]) not in missing_ids
    ]
    if not active_placements:
        raise FileNotFoundError(
            f"{style} has no available building GLBs to assemble in {buildings_dir}"
        )
    if missing_ids:
        print(
            f"Skipping {len(missing_ids)} missing building GLBs: {missing_ids}",
            flush=True,
        )
    cache_key = scene_cache_key(
        args,
        style,
        placement_data,
        active_placements,
        missing_ids,
        buildings_dir,
    )
    if blend_path.exists() and preview_path.exists() and not args.overwrite:
        if report_cache_key(report_path) == cache_key:
            print(f"Keeping current assembled scene: {blend_path}", flush=True)
            return
        print(
            f"Existing scene cache is stale; rebuilding: {blend_path}",
            flush=True,
        )
    scene_dir.mkdir(parents=True, exist_ok=True)

    clear_blender_scene(bpy)
    environment_report = (
        create_environment(bpy, args, style)
        if not args.no_environment
        else {
            "surface_cells": {},
            "semantic_pixels": {},
            "props": {},
            "terrain": "disabled",
            "sky": "disabled",
        }
    )
    buildings_collection = ensure_collection(bpy, "building")
    assembly_records = []

    for index, placement in enumerate(active_placements, start=1):
        building_id = placement["building_id"]
        glb_path = (
            buildings_dir / f"building_{building_id}" / "building.glb"
        )
        building, dimensions = import_joined_mesh(
            bpy,
            glb_path,
            f"building_{building_id}",
        )
        for collection in list(building.users_collection):
            collection.objects.unlink(building)
        buildings_collection.objects.link(building)

        asset_x, asset_y, asset_z = dimensions
        target_w = max(float(placement["footprint_w"]), 1.0)
        target_h = max(float(placement["footprint_h"]), 1.0)
        scale_default = min(target_w / asset_x, target_h / asset_y)
        scale_rotated = min(target_w / asset_y, target_h / asset_x)
        rotate_90 = scale_rotated > scale_default
        horizontal_scale = max(scale_default, scale_rotated) * args.footprint_fill
        if args.building_scale_mode == "uniform":
            vertical_scale = horizontal_scale
            actual_height = asset_z * horizontal_scale
        else:
            vertical_scale = (
                max(float(placement["height"]) * args.height_scale, 0.1)
                / asset_z
            )
            actual_height = asset_z * vertical_scale
        rotation_deg = -float(placement["angle_deg_image"])
        if rotate_90:
            rotation_deg += 90.0

        building.scale = (
            horizontal_scale,
            horizontal_scale,
            vertical_scale,
        )
        # glTF imports default to QUATERNION rotation mode. Assigning only
        # rotation_euler while leaving that mode unchanged updates the displayed
        # Euler values but not the object's transform matrix.
        building.rotation_mode = "XYZ"
        building.rotation_euler = (
            0.0,
            0.0,
            math.radians(rotation_deg),
        )
        location_x = float(placement["cx"])
        location_y = -float(placement["cy"])
        building.location = (
            location_x,
            location_y,
            terrain_height(location_x, location_y) + 0.10,
        )
        assembly_records.append(
            {
                "building_id": building_id,
                "source_glb": str(glb_path),
                "location": list(building.location),
                "rotation_z_deg": rotation_deg,
                "scale": list(building.scale),
                "source_dimensions": list(dimensions),
                "target_footprint": [target_w, target_h],
                "target_height": float(placement["height"]) * args.height_scale,
                "actual_height": actual_height,
                "scale_mode": args.building_scale_mode,
            }
        )
        print(
            f"[{index}/{len(active_placements)}] Placed building_{building_id}",
            flush=True,
        )

    width = float(placement_data["layout_width"])
    height = float(placement_data["layout_height"])
    max_height = max(record["actual_height"] for record in assembly_records)
    setup_camera_and_lighting(bpy, width, height, max_height)
    configure_render(bpy, preview_path, args.resolution)

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.render.render(write_still=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "style": style,
                "case_name": args.case_name,
                "scene_cache_version": SCENE_CACHE_VERSION,
                "scene_cache_key": cache_key,
                "building_count": len(assembly_records),
                "layout_building_count": len(placements),
                "skipped_building_count": len(missing_ids),
                "skipped_building_ids": missing_ids,
                "building_scale_mode": args.building_scale_mode,
                "assembly_config": {
                    "footprint_fill": args.footprint_fill,
                    "height_scale": args.height_scale,
                    "building_scale_mode": args.building_scale_mode,
                    "surface_cell_size": args.surface_cell_size,
                    "resolution": args.resolution,
                    "no_environment": args.no_environment,
                    "no_props": args.no_props,
                },
                "environment": environment_report,
                "blend_path": str(blend_path),
                "preview_path": str(preview_path),
                "buildings": assembly_records,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved Blender scene: {blend_path}", flush=True)
    print(f"Saved preview: {preview_path}", flush=True)


def child_command(
    interpreter: Path,
    args: argparse.Namespace,
    worker: str,
    style: str | None = None,
) -> list[str]:
    command = [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--case-name",
        args.case_name,
        "--layout-path",
        str(args.layout_path),
        "--depth-path",
        str(args.depth_path),
        "--layout-palette",
        args.layout_palette,
        "--placements-file",
        str(args.placements_file),
        "--footprint-fill",
        str(args.footprint_fill),
        "--height-scale",
        str(args.height_scale),
        "--building-scale-mode",
        args.building_scale_mode,
        "--surface-cell-size",
        str(args.surface_cell_size),
        "--resolution",
        str(args.resolution),
    ]
    if style is not None:
        command.extend(["--style", style])
    if args.assets_name:
        command.extend(["--assets-name", args.assets_name])
    if args.overwrite:
        command.append("--overwrite")
    if args.no_environment:
        command.append("--no-environment")
    if args.no_props:
        command.append("--no-props")
    if args.allow_missing_buildings:
        command.append("--allow-missing-buildings")
    return command


def parent_main(args: argparse.Namespace) -> None:
    styles = available_styles(args.case_name)
    if args.style == "all":
        selected_styles = styles or (sorted(STYLE_PALETTES) if args.dry_run else [])
    else:
        style_is_plannable = args.dry_run and args.style in STYLE_PALETTES
        if args.style not in styles and not style_is_plannable:
            raise ValueError(
                f"Unknown style {args.style!r}; available: {', '.join(styles)}"
            )
        selected_styles = [args.style]
    if args.style == "all" and args.assets_name:
        raise ValueError("--assets-name cannot be combined with --style all.")
    if not 0 < args.footprint_fill <= 1.0:
        raise ValueError("--footprint-fill must be in the range (0, 1].")
    if args.height_scale <= 0:
        raise ValueError("--height-scale must be positive.")
    if args.surface_cell_size <= 0:
        raise ValueError("--surface-cell-size must be positive.")

    print(f"Selected styles: {', '.join(selected_styles)}")
    print(f"Placements: {args.placements_file}")
    if args.dry_run:
        print(
            "Dry run passed; layout extraction and Blender were not started.",
            flush=True,
        )
        return

    subprocess.run(
        child_command(SYNCITY_PYTHON, args, "prepare"),
        cwd=PROJECT_ROOT,
        check=True,
    )
    for style in selected_styles:
        subprocess.run(
            child_command(BLENDER_PYTHON, args, "blender", style=style),
            cwd=PROJECT_ROOT,
            check=True,
        )
    print("Scene assembly completed.", flush=True)


def main() -> None:
    args = parse_args()
    args.layout_path = args.layout_path.expanduser().resolve()
    args.depth_path = args.depth_path.expanduser().resolve()
    args.placements_file = (
        args.placements_file.expanduser().resolve()
        if args.placements_file
        else placement_file_for_case(args.case_name).resolve()
    )

    if args.worker == "prepare":
        prepare_placements(args)
    elif args.worker == "blender":
        assemble_blender_scene(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
