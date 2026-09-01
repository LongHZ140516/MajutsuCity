#!/usr/bin/env python3
"""Run the unchanged upstream layout demo with an E2E-local config overlay."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DEMO = PROJECT_ROOT / "layout_gen" / "demo.py"
E2E_LONGCLIP_TARGET = "runtime.longclip_embedder.E2ELongCLIPEmbedder"


def option_value(arguments: list[str], option: str) -> tuple[str | None, list[str]]:
    remaining = []
    value = None
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            value = arguments[index + 1]
            index += 2
            continue
        if item.startswith(option + "="):
            value = item.split("=", maxsplit=1)[1]
            index += 1
            continue
        remaining.append(item)
        index += 1
    return value, remaining


def replace_option(arguments: list[str], option: str, value: str) -> list[str]:
    result = []
    replaced = False
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            result.extend((option, value))
            replaced = True
            index += 2
            continue
        if item.startswith(option + "="):
            result.extend((option, value))
            replaced = True
            index += 1
            continue
        result.append(item)
        index += 1
    if not replaced:
        result.extend((option, value))
    return result


def main() -> None:
    longclip_path, forwarded = option_value(sys.argv[1:], "--longclip_path")
    temporary_config: Path | None = None
    try:
        if longclip_path is not None:
            config_path, _ = option_value(forwarded, "--config_path")
            if config_path is None:
                raise ValueError("--longclip_path requires --config_path")
            checkpoint = Path(longclip_path).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"LongCLIP checkpoint not found: {checkpoint}")
            config = OmegaConf.load(Path(config_path).expanduser().resolve())
            condition = config.model.params.cond_stage_config
            condition.target = E2E_LONGCLIP_TARGET
            condition.params.version = str(checkpoint)
            condition.params.context_dim = 1024
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".yaml",
                prefix="majutsucity_layout_",
                delete=False,
            ) as file:
                temporary_config = Path(file.name)
                OmegaConf.save(config=config, f=file)
            forwarded = replace_option(
                forwarded, "--config_path", str(temporary_config)
            )

        subprocess.run(
            [sys.executable, str(UPSTREAM_DEMO), *forwarded],
            cwd=PROJECT_ROOT,
            check=True,
        )
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
