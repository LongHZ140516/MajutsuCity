# Unified runtime environment

The pipeline uses one Python 3.10 environment named `majutsucity`.
`environment.yml` combines the Hunyuan3D, Qwen Image, Syncity, layout and common
pipeline dependencies with pinned top-level versions.

For a fresh Linux installation with a CUDA 12.4-capable NVIDIA driver, create
the public environment directly from the repository root:

```bash
conda env create -f environment.yml
conda activate majutsucity
```

`environment.yml` installs the curated Hunyuan runtime, the Qwen /
Syncity overlay, the common pipeline packages and the optional layout packages.
It does not download any model weights or external Hunyuan3D source trees.

For an existing compatible Python 3.10 environment, the equivalent unified
Python dependency entry point is:

```bash
python -m pip install -r requirements.txt
```

The root file includes the Hunyuan3D runtime, framework overlay, common
pipeline and layout dependency groups under `requirements/` in installation
order.

Place the upstream Hunyuan3D-Omni and Hunyuan3D-2.1 repositories at
`external/Hunyuan3D-Omni` and `external/Hunyuan3D-2.1`. The Omni source provides
point-controlled shape generation; the 2.1 source provides PBR painting. Compile
the two 2.1 paint extensions with the active environment:

```bash
mkdir -p external
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git external/Hunyuan3D-Omni
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git external/Hunyuan3D-2.1

cd external/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
python -m pip install -e .

cd ../DifferentiableRenderer
bash compile_mesh_painter.sh

cd ../../../..
```

The build requires a working CUDA toolkit and C++ compiler compatible with the
installed PyTorch wheel. The expected products are
`custom_rasterizer_kernel*.so` and `mesh_inpaint_processor*.so`; the preflight
tool checks both without loading model weights.

The pinned baseline is:

- Python 3.10
- PyTorch 2.6.0 + CUDA 12.4
- torchvision 0.21.0 + CUDA 12.4
- Hunyuan3D-Omni shape, Hunyuan3D-2.1 paint extensions, and `bpy` 4.0
- Diffusers 0.37.1 and Transformers 4.57.3
- PEFT 0.17.0 for the Texture/Skybox LoRA loaders
- PyVista 0.46.5 and VTK 9.5.2

PyTorch 2.6 or newer is required by Transformers 4.57 when Hunyuan paint loads
legacy PyTorch checkpoint files; this also matches the validated Qwen runtime.

Use `environment.yml` and compile the two extensions as shown above. The same
interpreter is used for controller, layout, rendering, Qwen, Hunyuan and
Blender stages in `configs/paths.example.yaml`.

The optional layout stage additionally uses the local MajutsuCity layout and
height checkpoints plus `models/LongCLIP-L/longclip-L.pt`. The layout
requirements file installs the OpenCLIP tokenizer vocabulary used by the
bundled LongCLIP implementation, so no tokenizer asset is downloaded at
runtime.

Machine-local environment/model directories can be linked below `envs/`,
`models/` and `external/`; all three locations are ignored by Git.

Off-screen building rendering also requires the system `Xvfb` executable. The
pipeline sets a writable temporary Matplotlib cache directory automatically.

After configuring local model paths, validate the installation before starting
an expensive generation run:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
python scripts/preflight.py --config configs/paths.local.yaml --scope full
```

Use `--scope 3d` or `--scope layout` for a smaller component check, and
`--skip-gpu` when validating only files and Python dependencies. A failed check
returns a non-zero exit status. `--json-output <path>` optionally writes a
machine-readable report.
