import gc
import math
import os
from PIL import Image
from typing import Optional

from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline
)
from diffusers.models import QwenImageTransformer2DModel
import torch

class Qwen_Image:
    def __init__(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        local_files_only: bool = False,
        device_map: Optional[str] = None,
    ):
        self.device = device
        model_basename = os.path.basename(os.path.normpath(model_name))

        # Select the pipeline from either a Hub ID or a local model directory.
        if model_name == "Qwen/Qwen-Image-Edit" or model_basename == "Qwen-Image-Edit":
            self.pipe_cls = QwenImageEditPipeline
            print("Note: You are using the image-edit model. If you want to do text-to-image generation, please use the text-to-image model.")
        elif model_name == "Qwen/Qwen-Image-Edit-2509" or model_basename == "Qwen-Image-Edit-2509":
            self.pipe_cls = QwenImageEditPlusPipeline
            print("Note: You are using the image-edit-plus model. If you want to do text-to-image generation, please use the text-to-image model.")
        elif model_name == "Qwen/Qwen-Image" or model_basename == "Qwen-Image":
            self.pipe_cls = DiffusionPipeline
            print("Note: You are using the text-to-image model. If you want to do image editing, please use the image-edit model.")
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        # load lora weights
        if lora_path is not None:
            self.load_lora_weights(model_name, lora_path)
        else:
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "local_files_only": local_files_only,
            }
            if device_map:
                load_kwargs["device_map"] = device_map

            self.pipe = self.pipe_cls.from_pretrained(model_name, **load_kwargs)
            if not device_map:
                self.pipe = self.pipe.to(device)

    def load_lora_weights(self, model_name: str, lora_path: str):
        assert os.path.exists(lora_path), f"Lora path {lora_path} does not exist" 

        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
            
        else:
            torch_dtype = torch.float32
            device = "cpu"

        model = QwenImageTransformer2DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch_dtype, device=device
        )

        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        self.pipe = self.pipe_cls.from_pretrained(
            model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
        ).to(device)
        self.pipe.load_lora_weights(lora_path)
    
    def memory_optimization(self, enable_xformers: bool = False, enable_sequential_cpu_offload: bool = True, enable_model_cpu_offload: bool = False, enable_cpu_offload: bool = False):
        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except Exception as e:
                print(f"Warning: Could not enable xformers: {e}")

        if enable_sequential_cpu_offload:
            try:
                self.pipe.enable_sequential_cpu_offload()
                print("Enabled sequential CPU offload")
            except Exception as e:
                print(f"Warning: Could not enable sequential CPU offload: {e}")
        elif enable_model_cpu_offload:
            try:
                self.pipe.enable_model_cpu_offload()
                print("Enabled model CPU offload")
            except Exception as e:
                print(f"Warning: Could not enable model CPU offload: {e}")
        elif enable_cpu_offload:
            try:
                self.pipe.enable_offload_submodels()
                print("Enabled CPU offload")
            except Exception as e:
                print(f"Warning: Could not enable CPU offload: {e}")
    
    def generate(self, 
        prompt: str, 
        negative_prompt: str,
        seed: int = 42,
        image: Optional[Image.Image] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 8, 
        true_cfg_scale: float = 1.0,
        memory_optimize: bool = True):
        
        # if memory_optimize:
        #     self.memory_optimization()
        input_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "generator": torch.Generator(device=self.device).manual_seed(seed),
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
        }
        if width is not None and height is not None:
            input_args["width"] = width
            input_args["height"] = height
        if image is not None:
            input_args["image"] = image

        # Use the correct method for image generation depending on the pipeline class
        if hasattr(self.pipe, "generate"):
            result = self.pipe.generate(**input_args)
        else:
            result = self.pipe(**input_args)
            
        image = result.images[0]
        return image

    def unload(self):
        """Release all pipeline weights before loading the background remover."""
        if getattr(self, "pipe", None) is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
