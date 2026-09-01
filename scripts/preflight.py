#!/usr/bin/env python3
"""Validate a MajutsuCity installation without loading model weights."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_CONFIG = PROJECT_ROOT / "configs" / "paths.local.yaml"
DEFAULT_EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "paths.example.yaml"


@dataclass
class Check:
    status: str
    group: str
    name: str
    detail: str


class Reporter:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, status: str, group: str, name: str, detail: str) -> None:
        self.checks.append(Check(status, group, name, detail))

    def passed(self, group: str, name: str, detail: str) -> None:
        self.add("PASS", group, name, detail)

    def warning(self, group: str, name: str, detail: str) -> None:
        self.add("WARN", group, name, detail)

    def failed(self, group: str, name: str, detail: str) -> None:
        self.add("FAIL", group, name, detail)

    def render(self) -> None:
        for check in self.checks:
            print(
                f"[{check.status:4s}] {check.group:12s} {check.name}: {check.detail}",
                flush=True,
            )
        counts = {
            status: sum(check.status == status for check in self.checks)
            for status in ("PASS", "WARN", "FAIL")
        }
        print(
            "\nPreflight summary: "
            f"{counts['PASS']} passed, {counts['WARN']} warnings, "
            f"{counts['FAIL']} failed.",
            flush=True,
        )


PACKAGE_MODULES = {
    "accelerate": "accelerate",
    "bpy": "bpy",
    "diffusers": "diffusers",
    "einops": "einops",
    "ftfy": "ftfy",
    "kornia": "kornia",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "open-clip-torch": "open_clip",
    "opencv-python-headless": "cv2",
    "peft": "peft",
    "Pillow": "PIL",
    "pymeshlab": "pymeshlab",
    "pytorch-lightning": "pytorch_lightning",
    "PyYAML": "yaml",
    "realesrgan": "realesrgan",
    "scikit-image": "skimage",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "trimesh": "trimesh",
    "vtk": "vtk",
    "pyvista": "pyvista",
    "xatlas": "xatlas",
}

ROLE_PACKAGES = {
    "controller": {"PyYAML"},
    "layout": {
        "einops",
        "ftfy",
        "numpy",
        "omegaconf",
        "open-clip-torch",
        "opencv-python-headless",
        "Pillow",
        "pytorch-lightning",
        "PyYAML",
        "torch",
        "torchvision",
    },
    "render": {
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "pyvista",
        "PyYAML",
        "scikit-image",
        "vtk",
    },
    "qwen": {
        "accelerate",
        "diffusers",
        "kornia",
        "peft",
        "Pillow",
        "torch",
        "torchvision",
        "transformers",
    },
    "hunyuan": {
        "accelerate",
        "bpy",
        "diffusers",
        "Pillow",
        "pymeshlab",
        "realesrgan",
        "torch",
        "torchvision",
        "transformers",
        "trimesh",
        "xatlas",
    },
    "blender": {"bpy"},
}

VALIDATED_VERSION_PREFIXES = {
    "accelerate": "1.13.",
    "bpy": "4.0.",
    "diffusers": "0.37.",
    "numpy": "1.24.",
    "open-clip-torch": "2.24.",
    "opencv-python-headless": "4.11.",
    "Pillow": "11.",
    "peft": "0.17.",
    "pytorch-lightning": "1.9.",
    "PyYAML": "6.0.",
    "scikit-image": "0.24.",
    "torch": "2.6.",
    "torchvision": "0.21.",
    "transformers": "4.57.",
    "trimesh": "4.4.",
    "vtk": "9.5.",
    "pyvista": "0.46.",
    "xatlas": "0.0.9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check MajutsuCity configuration, interpreters, dependencies, local "
            "weights, Hunyuan extensions, Xvfb and configured GPUs without "
            "loading model weights."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path configuration. Defaults to configs/paths.local.yaml when it "
            "exists, otherwise configs/paths.example.yaml."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("full", "3d", "layout"),
        default="full",
        help="Validate the full pipeline, only 3D stages, or only layout stages.",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip NVIDIA driver and PyTorch CUDA checks.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optionally save the complete check report as JSON.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def resolve_config_path(value: Path | None) -> Path:
    if value is not None:
        return project_path(value)
    return DEFAULT_LOCAL_CONFIG if DEFAULT_LOCAL_CONFIG.is_file() else DEFAULT_EXAMPLE_CONFIG


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def load_config(path: Path, reporter: Reporter) -> dict[str, Any] | None:
    if not path.is_file():
        reporter.failed("configuration", "config file", f"not found: {path}")
        return None
    if importlib.util.find_spec("yaml") is None:
        reporter.failed(
            "configuration",
            "PyYAML",
            "not installed in the Python used to launch preflight",
        )
        return None
    import yaml

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as error:
        reporter.failed("configuration", "config syntax", str(error))
        return None
    if not isinstance(config, dict):
        reporter.failed("configuration", "config structure", "expected a YAML mapping")
        return None
    if config.get("version") != 1:
        reporter.failed(
            "configuration", "config version", f"expected 1, got {config.get('version')!r}"
        )
        return None
    reporter.passed("configuration", "config file", str(path))
    return config


def config_path(
    config: dict[str, Any], reporter: Reporter, label: str, *keys: str
) -> Path | None:
    try:
        value = nested(config, *keys)
    except KeyError as error:
        reporter.failed("configuration", label, f"missing key: {error.args[0]}")
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        reporter.failed("configuration", label, "path must be a non-empty string")
        return None
    return project_path(value)


def check_file(
    reporter: Reporter, group: str, label: str, path: Path | None
) -> None:
    if path is None:
        return
    if not path.is_file():
        reporter.failed(group, label, f"file not found: {path}")
    elif path.stat().st_size == 0:
        reporter.failed(group, label, f"file is empty: {path}")
    else:
        reporter.passed(group, label, f"{path} ({path.stat().st_size:,} bytes)")


def check_directory(
    reporter: Reporter,
    group: str,
    label: str,
    path: Path | None,
    markers: tuple[str, ...] = (),
) -> None:
    if path is None:
        return
    if not path.is_dir():
        reporter.failed(group, label, f"directory not found: {path}")
        return
    missing = [marker for marker in markers if not (path / marker).exists()]
    if missing:
        reporter.failed(
            group, label, f"incomplete at {path}; missing: {', '.join(missing)}"
        )
    else:
        reporter.passed(group, label, str(path))


def interpreter_roles(scope: str) -> tuple[str, ...]:
    roles = ["controller"]
    if scope in ("full", "layout"):
        roles.append("layout")
    if scope in ("full", "3d"):
        roles.extend(("render", "qwen", "hunyuan", "blender"))
    return tuple(roles)


def probe_environment(
    interpreter: Path, packages: set[str]
) -> tuple[dict[str, Any] | None, str | None]:
    probe = r"""
import importlib.metadata
import importlib.util
import json
import sys

requested = json.loads(sys.argv[1])
result = {"python": sys.version.split()[0], "packages": {}}
for distribution, module in requested.items():
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        available = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    result["packages"][distribution] = {"version": version, "importable": available}
print(json.dumps(result, sort_keys=True))
"""
    mapping = {name: PACKAGE_MODULES[name] for name in sorted(packages)}
    completed = subprocess.run(
        [str(interpreter), "-c", probe, json.dumps(mapping)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return None, detail or f"probe exited with code {completed.returncode}"
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1]), None
    except (json.JSONDecodeError, IndexError) as error:
        return None, f"invalid probe output: {error}"


def check_interpreters(
    config: dict[str, Any], scope: str, reporter: Reporter
) -> dict[Path, set[str]]:
    environments: dict[Path, set[str]] = {}
    labels: dict[Path, list[str]] = {}
    for role in interpreter_roles(scope):
        path = config_path(
            config,
            reporter,
            f"{role} interpreter",
            "paths",
            "interpreters",
            role,
        )
        if path is None:
            continue
        if not path.is_file() or not os.access(path, os.X_OK):
            reporter.failed("interpreters", role, f"not executable: {path}")
            continue
        reporter.passed("interpreters", role, str(path))
        environments.setdefault(path, set()).update(ROLE_PACKAGES[role])
        labels.setdefault(path, []).append(role)

    for path, packages in environments.items():
        result, error = probe_environment(path, packages)
        name = "+".join(labels[path])
        if error or result is None:
            reporter.failed("dependencies", name, error or "unknown probe failure")
            continue
        version = str(result["python"])
        if version.startswith("3.10."):
            reporter.passed("dependencies", f"Python ({name})", version)
        else:
            reporter.failed(
                "dependencies", f"Python ({name})", f"expected 3.10.x, got {version}"
            )
        for package, data in result["packages"].items():
            if not data["importable"] or data["version"] is None:
                reporter.failed(
                    "dependencies", package, f"missing from interpreter: {path}"
                )
                continue
            installed = str(data["version"])
            expected = VALIDATED_VERSION_PREFIXES.get(package)
            if expected and not installed.startswith(expected):
                reporter.warning(
                    "dependencies",
                    package,
                    f"installed {installed}; validated baseline is {expected}x",
                )
            else:
                reporter.passed("dependencies", package, installed)
    return environments


def check_pip(environments: dict[Path, set[str]], reporter: Reporter) -> None:
    tolerated = (
        "bpy 4.0.0 is not supported on this platform",
        "ninja 1.11.1.1 is not supported on this platform",
    )
    for interpreter in environments:
        completed = subprocess.run(
            [str(interpreter), "-m", "pip", "check"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        unexpected = [line for line in lines if line not in tolerated]
        if completed.returncode == 0:
            reporter.passed("dependencies", "pip check", str(interpreter))
        elif lines and not unexpected:
            reporter.warning(
                "dependencies",
                "pip metadata",
                "known bpy/ninja platform-metadata warnings; runtime is checked separately",
            )
        else:
            detail = "; ".join(unexpected) or completed.stderr.strip()
            reporter.failed("dependencies", "pip check", detail or "unknown failure")


def check_layout_assets(config: dict[str, Any], reporter: Reporter) -> None:
    checks = (
        ("LongCLIP-L", ("paths", "models", "longclip")),
        ("layout checkpoint", ("layout_generation", "layout_checkpoint")),
        ("height checkpoint", ("layout_generation", "height_checkpoint")),
        ("layout model config", ("layout_generation", "layout_model_config")),
        ("height model config", ("layout_generation", "height_model_config")),
    )
    for label, keys in checks:
        check_file(
            reporter,
            "layout assets",
            label,
            config_path(config, reporter, label, *keys),
        )


def check_3d_assets(config: dict[str, Any], reporter: Reporter) -> None:
    model_specs = (
        (
            "Qwen Image Edit",
            ("paths", "models", "qwen_image_edit"),
            ("model_index.json", "transformer", "text_encoder", "vae"),
        ),
        (
            "Qwen Image",
            ("paths", "models", "qwen_image"),
            ("model_index.json", "transformer", "text_encoder", "vae"),
        ),
        (
            "RMBG-2.0",
            ("paths", "models", "rmbg"),
            ("config.json", "model.safetensors"),
        ),
        (
            "Hunyuan3D-2.1 weights",
            ("paths", "models", "hunyuan3d"),
            (
                "hunyuan3d-dit-v2-1/model.fp16.ckpt",
                "hunyuan3d-vae-v2-1/model.fp16.ckpt",
                "hunyuan3d-paintpbr-v2-1/model_index.json",
            ),
        ),
        (
            "DINOv2 Giant",
            ("paths", "models", "dinov2"),
            ("config.json", "model.safetensors"),
        ),
    )
    for label, keys, markers in model_specs:
        check_directory(
            reporter,
            "3D assets",
            label,
            config_path(config, reporter, label, *keys),
            markers,
        )

    for label, keys in (
        ("Texture LoRA", ("paths", "models", "texture_lora")),
        ("Skybox LoRA", ("paths", "models", "skybox_lora")),
    ):
        check_file(
            reporter,
            "3D assets",
            label,
            config_path(config, reporter, label, *keys),
        )

    repo = config_path(
        config,
        reporter,
        "Hunyuan source",
        "paths",
        "external",
        "hunyuan3d_repo",
    )
    check_directory(
        reporter,
        "3D runtime",
        "Hunyuan source",
        repo,
        (
            "hy3dshape/hy3dshape",
            "hy3dpaint/textureGenPipeline.py",
            "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml",
            "hy3dpaint/ckpt/RealESRGAN_x4plus.pth",
        ),
    )
    if repo and repo.is_dir():
        rasterizers = tuple(
            (repo / "hy3dpaint" / "custom_rasterizer").glob(
                "custom_rasterizer_kernel*.so"
            )
        )
        mesh_processors = tuple(
            (repo / "hy3dpaint" / "DifferentiableRenderer").glob(
                "mesh_inpaint_processor*.so"
            )
        )
        if rasterizers:
            reporter.passed(
                "3D runtime", "custom rasterizer", str(rasterizers[0])
            )
        else:
            reporter.failed(
                "3D runtime",
                "custom rasterizer",
                "compiled custom_rasterizer_kernel*.so not found",
            )
        if mesh_processors:
            reporter.passed(
                "3D runtime", "mesh processor", str(mesh_processors[0])
            )
        else:
            reporter.failed(
                "3D runtime",
                "mesh processor",
                "compiled mesh_inpaint_processor*.so not found",
            )

    xvfb = shutil.which("Xvfb")
    if xvfb:
        reporter.passed("3D runtime", "Xvfb", xvfb)
    else:
        reporter.failed(
            "3D runtime", "Xvfb", "required for off-screen PyVista rendering"
        )


def parse_gpu_configuration(
    config: dict[str, Any], reporter: Reporter
) -> set[int] | None:
    try:
        qwen_spec = str(nested(config, "runtime", "qwen_gpu_groups"))
        hunyuan_spec = str(nested(config, "runtime", "hunyuan_gpus"))
    except KeyError as error:
        reporter.failed("GPU", "runtime config", f"missing key: {error.args[0]}")
        return None
    requested: set[int] = set()
    used_qwen: set[int] = set()
    try:
        groups = [group.strip() for group in qwen_spec.split(";") if group.strip()]
        if not groups:
            raise ValueError("qwen_gpu_groups is empty")
        for group in groups:
            ids = [int(item.strip()) for item in group.split(",") if item.strip()]
            if len(ids) != 2 or len(set(ids)) != 2 or any(item < 0 for item in ids):
                raise ValueError(f"Qwen group must contain two distinct IDs: {group!r}")
            overlap = used_qwen.intersection(ids)
            if overlap:
                raise ValueError(f"Qwen GPU IDs reused across groups: {sorted(overlap)}")
            used_qwen.update(ids)
        hunyuan_ids = [
            int(item.strip()) for item in hunyuan_spec.split(",") if item.strip()
        ]
        if not hunyuan_ids or len(hunyuan_ids) != len(set(hunyuan_ids)):
            raise ValueError("hunyuan_gpus must contain distinct GPU IDs")
        if any(item < 0 for item in hunyuan_ids):
            raise ValueError("GPU IDs must be non-negative")
    except ValueError as error:
        reporter.failed("GPU", "runtime config", str(error))
        return None
    requested.update(used_qwen)
    requested.update(hunyuan_ids)
    reporter.passed(
        "GPU",
        "runtime config",
        f"Qwen={qwen_spec}; Hunyuan={hunyuan_spec}; requested={sorted(requested)}",
    )
    return requested


def check_gpus(
    config: dict[str, Any], controller: Path | None, reporter: Reporter
) -> None:
    requested = parse_gpu_configuration(config, reporter)
    executable = shutil.which("nvidia-smi")
    if not executable:
        reporter.failed("GPU", "nvidia-smi", "executable not found")
        return
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        reporter.failed(
            "GPU", "NVIDIA driver", (completed.stderr or completed.stdout).strip()
        )
        return
    available: set[int] = set()
    descriptions = []
    try:
        for line in completed.stdout.splitlines():
            index_text, name, memory = [item.strip() for item in line.split(",", 2)]
            index = int(index_text)
            available.add(index)
            descriptions.append(f"{index}:{name} ({memory} MiB)")
    except (ValueError, IndexError) as error:
        reporter.failed("GPU", "nvidia-smi output", str(error))
        return
    reporter.passed("GPU", "NVIDIA driver", "; ".join(descriptions))
    if requested is not None:
        missing = requested - available
        if missing:
            reporter.failed("GPU", "configured IDs", f"not visible: {sorted(missing)}")
        else:
            reporter.passed("GPU", "configured IDs", str(sorted(requested)))

    if controller is None:
        reporter.failed("GPU", "PyTorch CUDA", "controller interpreter unavailable")
        return
    probe = (
        "import json,torch; "
        "print(json.dumps({'available':torch.cuda.is_available(),"
        "'count':torch.cuda.device_count(),'torch':torch.__version__,"
        "'cuda':torch.version.cuda}))"
    )
    torch_probe = subprocess.run(
        [str(controller), "-c", probe],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if torch_probe.returncode != 0:
        reporter.failed(
            "GPU", "PyTorch CUDA", (torch_probe.stderr or torch_probe.stdout).strip()
        )
        return
    try:
        data = json.loads(torch_probe.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        reporter.failed("GPU", "PyTorch CUDA", f"invalid probe output: {error}")
        return
    if not data["available"]:
        reporter.failed("GPU", "PyTorch CUDA", f"not available; probe={data}")
    elif requested is not None and data["count"] <= max(requested, default=-1):
        reporter.failed(
            "GPU",
            "PyTorch CUDA",
            f"device_count={data['count']} does not cover IDs {sorted(requested)}",
        )
    else:
        reporter.passed(
            "GPU",
            "PyTorch CUDA",
            f"torch={data['torch']}, CUDA={data['cuda']}, devices={data['count']}",
        )


def check_output(reporter: Reporter) -> None:
    output = PROJECT_ROOT / "output"
    target = output if output.exists() else PROJECT_ROOT
    if target.is_dir() and os.access(target, os.W_OK):
        reporter.passed("filesystem", "output writable", str(output))
    else:
        reporter.failed("filesystem", "output writable", str(output))


def main() -> None:
    args = parse_args()
    reporter = Reporter()
    selected_config_path = resolve_config_path(args.config)
    config = load_config(selected_config_path, reporter)
    environments: dict[Path, set[str]] = {}
    if config is not None:
        environments = check_interpreters(config, args.scope, reporter)
        check_pip(environments, reporter)
        if args.scope in ("full", "layout"):
            check_layout_assets(config, reporter)
        if args.scope in ("full", "3d"):
            check_3d_assets(config, reporter)
        if args.skip_gpu:
            reporter.warning("GPU", "GPU checks", "skipped by --skip-gpu")
        else:
            controller = config_path(
                config,
                reporter,
                "controller interpreter",
                "paths",
                "interpreters",
                "controller",
            )
            check_gpus(config, controller, reporter)
    check_output(reporter)
    reporter.render()

    if args.json_output is not None:
        json_path = project_path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(selected_config_path),
            "scope": args.scope,
            "checks": [asdict(check) for check in reporter.checks],
            "summary": {
                status.lower(): sum(
                    check.status == status for check in reporter.checks
                )
                for status in ("PASS", "WARN", "FAIL")
            },
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {json_path}", flush=True)

    if any(check.status == "FAIL" for check in reporter.checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
