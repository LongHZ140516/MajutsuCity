# ControlNet Inference Demo

This repository contains a standalone inference script (`demo.py`) for generating images using trained ControlNet checkpoints. The script is streamlined for single-image generation and supports two primary tasks in the pipeline:
1. **Layout Generation**: Generating a semantic layout based purely on a text prompt.
2. **Heightmap / Result Generation**: Generating a heightmap or realistic result based on an existing semantic layout image and a text prompt.

## Prerequisites

Ensure you have the required environment set up, including:
- PyTorch
- OpenCV (`cv2`)
- Pillow (`PIL`)
- PyTorch Lightning
- Your ControlNet environment dependencies (e.g., `share`, `cldm`)

## Usage

The script is executed via the command line using `argparse`. Since it is optimized for single-image generation, it only requires 1 GPU. SLURM cluster using `srun`.

### Mode 1: Layout Generation
In this mode, the model generates a layout from scratch based on a detailed text prompt. 

**Key Parameter:** `--use_layout_condition False`

**Example Command:**
```bash
python demo.py \
    --use_layout_condition False \
    --config_path /path/to/models/cldm_v21.yaml \
    --model_path /path/to/your/layout_generation_model.ckpt \
    --result_dir ./results/demo_layout \
    --text_prompt "The road network (red) features a main east-west artery crossing the center, connected to parallel north-south roads in the west forming a grid, and curving roads in the east. Buildings (yellow) are dense, rectangular clusters in the west and irregular/L-shaped in the east. Vegetation (green) forms linear belts alongside the central water body and roads, with contiguous patches separating built areas. A central, north-south oriented river (blue) divides the image, crossed by a main road, with another water segment in the southeast." \
    --image_width 512 \
    --image_height 512 \
    --seed 42
```

### Mode 2: Heightmap / Result Generation
In this mode, the model uses an existing layout image as a structural constraint to generate the final output (e.g., a heightmap or satellite view). 

**Key Parameters:** - `--use_layout_condition True`
- `--control_image_path /path/to/input.png` (Required)

**Example Command:**
```bash
python demo.py \
    --use_layout_condition True \
    --control_image_path /path/to/your/generated_layout_map.png \
    --config_path /path/to/models/cldm_v21.yaml \
    --model_path /path/to/your/heightmap_generation_model.ckpt \
    --result_dir ./results/demo_heightmap \
    --text_prompt "A standard height map generate from the input segmentation map." \
    --image_width 512 \
    --image_height 512 \
    --seed 42
```

## Training and Evaluation

For detailed parameter configurations and execution instructions, please refer to the `train.sh` and `test.sh` scripts provided in this repository. 
(You can train on your own data, as long as the data format is organized according to [MajutsuDataset-Layout](https://huggingface.co/datasets/SereinH/MajutsuDataset/blob/main/layout.zip).)