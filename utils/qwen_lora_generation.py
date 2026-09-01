"""Shared local-only Qwen Image LoRA generation used by material entry points."""

from __future__ import annotations

import gc
from pathlib import Path


def generate_lora_image(
    *,
    model_path: Path,
    lora_path: Path,
    output_path: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    true_cfg_scale: float,
) -> None:
    """Load a local Qwen Image snapshot plus one LoRA and save one image."""
    import torch
    from diffusers import DiffusionPipeline

    model_path = model_path.expanduser().resolve()
    lora_path = lora_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen Image directory not found: {model_path}")
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    load_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if use_cuda and torch.cuda.device_count() > 1:
        load_kwargs["device_map"] = "balanced"

    print(f"Loading local Qwen Image: {model_path}", flush=True)
    pipe = DiffusionPipeline.from_pretrained(str(model_path), **load_kwargs)
    if use_cuda and "device_map" not in load_kwargs:
        pipe = pipe.to("cuda")
    elif not use_cuda:
        pipe = pipe.to("cpu")

    print(f"Loading LoRA: {lora_path}", flush=True)
    pipe.load_lora_weights(
        str(lora_path.parent),
        weight_name=lora_path.name,
        local_files_only=True,
    )
    generator_device = "cuda" if use_cuda else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=true_cfg_scale,
        generator=generator,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(f"Saved generated image: {output_path}", flush=True)

    del result
    del pipe
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
