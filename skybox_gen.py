import torch
import random
import os
from diffusers import DiffusionPipeline

model_name = "Qwen/Qwen-Image"
lora_path = './ckpt/texture/SkyBox_Lora.safetensors'  # replace with SKYBOX LoRA weights path
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# default
prompt = "A breathtaking fantasy anime sky: vibrant azure blue, majestic towering cumulonimbus clouds, anime style, skybox equirectangular 360 texture."
negative_prompt = "low quality, blurry, distorted, text, watermark"

def run_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading pipeline on {device}...")

    # load the base model
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    # load LoRA weights
    print("Loading LoRA weights...")
    pipe.load_lora_weights(lora_path)

    # generate a random seed for reproducibility
    width, height = 2048, 1024
    seed = random.randint(0, 2**30 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    # generate the image
    print(f"Generating image with seed: {seed}...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator
    )

    # save the generated image
    image = result.images[0]
    save_path = os.path.join(output_dir, "skybox_demo.png")
    image.save(save_path)
    print(f"Done! Image saved to: {save_path}")

if __name__ == "__main__":
    run_demo()