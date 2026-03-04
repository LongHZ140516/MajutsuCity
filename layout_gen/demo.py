import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything

# Import necessary modules for ControlNet
from share import *
from cldm.model import create_model, load_state_dict

def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet Single Image Inference Demo")
    
    # Core path parameters
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML model configuration file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained ckpt weight file")
    parser.add_argument("--result_dir", type=str, default="./demo_results", help="Directory to save the generated results")
    
    # Task differentiation logic: use_layout_condition
    parser.add_argument("--use_layout_condition", type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help="Set to True: Generate result/heightmap based on a layout (requires --control_image_path); Set to False: Generate the layout itself (uses an all-black condition image).")
    parser.add_argument("--control_image_path", type=str, default="", 
                        help="Path to the control condition image (e.g., an existing layout map). Required when use_layout_condition=True.")
    
    # Generation parameters
    parser.add_argument("--text_prompt", type=str, 
                        default="The road network (red) features a main east-west artery crossing the center, connected to parallel north-south roads in the west forming a grid, and curving roads in the east. Buildings (yellow) are dense, rectangular clusters in the west and irregular/L-shaped in the east. Vegetation (green) forms linear belts alongside the central water body and roads, with contiguous patches separating built areas. A central, north-south oriented river (blue) divides the image, crossed by a main road, with another water segment in the southeast.", 
                        help="Positive text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative text prompt")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=9.0, help="CFG Scale")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of sampling steps")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    seed_everything(args.seed, workers=True)

    print(f"Loading model from {args.model_path}...")
    # 1. Initialize the model and load weights
    model = create_model(args.config_path).cpu()
    model.load_state_dict(load_state_dict(args.model_path, location='cpu'), strict=False)
    model = model.cuda()
    model.eval()

    # 2. Process the control image (Hint) based on use_layout_condition
    if args.use_layout_condition:
        if not args.control_image_path or not os.path.exists(args.control_image_path):
            raise FileNotFoundError(f"Current mode is use_layout_condition=True, please provide a valid layout image path: {args.control_image_path}")
            
        print(f"Mode: Generate based on Layout. Reading control image from {args.control_image_path} ...")
        control_img = cv2.imread(args.control_image_path)
        control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2RGB)
        control_img = cv2.resize(control_img, (args.image_width, args.image_height))
        # Normalize to float32 in range [0, 1]
        control_img = control_img.astype(np.float32) / 255.0
    else:
        print("Mode: Generate Layout. Using a completely zero (black) image as the conditional control image...")
        # When generating a layout, no external structure map is needed; use a completely zero matrix
        control_img = np.zeros((args.image_height, args.image_width, 3), dtype=np.float32)

    # 3. Construct a Dummy Target (fake original image for placeholder to avoid KeyError)
    dummy_target = np.zeros((args.image_height, args.image_width, 3), dtype=np.float32)

    # 4. Construct the Batch dictionary to feed into the model
    batch = {
        "txt": [args.text_prompt],
        "hint": torch.from_numpy(control_img).unsqueeze(0).cuda(),
        "jpg": torch.from_numpy(dummy_target).unsqueeze(0).cuda(),
        "name": ["demo_inference"] 
    }

    # 5. Execute inference
    print(f"Generating image for prompt: '{args.text_prompt}'...")
    with torch.no_grad():
        images = model.log_images(
            batch,
            N=1,
            unconditional_guidance_scale=args.unconditional_guidance_scale,
            ddim_steps=args.ddim_steps
        )

        # 6. Save results
        for k, img_tensor in images.items():
            img_tensor = img_tensor.detach().cpu()
            img_tensor = torch.clamp(img_tensor, -1., 1.)
            img_tensor = (img_tensor + 1.0) / 2.0

            img_numpy = img_tensor.squeeze(0).transpose(0, 1).transpose(1, 2).numpy()
            img_numpy = (img_numpy * 255).astype(np.uint8)

            # Differentiate the saved filename based on use_layout_condition
            mode_str = "heightmap" if args.use_layout_condition else "layout"
            filename = f"demo_{mode_str}_{k}.png"
            save_path = os.path.join(args.result_dir, filename)
            Image.fromarray(img_numpy).save(save_path)
            print(f"Result saved: {save_path}")

    print("Demo inference completed!")

if __name__ == '__main__':
    main()