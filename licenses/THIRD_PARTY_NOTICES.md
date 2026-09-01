# Third-party notices

MajutsuCity includes source derived from third-party projects. The table below
records the corresponding upstream projects and licenses. Complete primary
license texts and copyright statements are collected in
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES); component-specific license files
and source-level attribution comments remain alongside the applicable code.

## Bundled source

| Local area | Upstream project | License |
| --- | --- | --- |
| `layout_gen/cldm/` and portions of `layout_gen/ldm/` | [ControlNet](https://github.com/lllyasviel/ControlNet) | Apache-2.0 |
| portions of `layout_gen/ldm/` | [Latent Diffusion](https://github.com/CompVis/latent-diffusion) | MIT |
| portions of `layout_gen/ldm/models/diffusion/` | [Taming Transformers](https://github.com/CompVis/taming-transformers) | MIT |
| diffusion utility code | [Guided Diffusion](https://github.com/openai/guided-diffusion) and [Improved Diffusion](https://github.com/openai/improved-diffusion) | MIT |
| `layout_gen/model/longclip.py` and related tokenizer files | [Long-CLIP](https://github.com/beichenzbc/Long-CLIP) | Apache-2.0 |
| layout-generation tokenizer vocabulary and runtime dependency | [OpenCLIP](https://github.com/mlfoundations/open_clip) | MIT |

Additional source files retain their original inline attribution links for
OpenAI CLIP, Diffusers, PyTorch and other credited implementations.

## Inherited annotator utilities

The optional `layout_gen/annotator/` tree contains third-party detector and
image-processing utilities inherited from the layout-generation codebase.
Applicable license files remain in their component directories. These
utilities are not imported by the public end-to-end inference entry point.

## Models and checkpoints

Model weights are not distributed in this repository. Model sources, local
placement conventions and their separate usage terms are listed in the
[model setup section of the main README](../README.md#model-setup). A software
license for this repository does not replace or modify the terms attached to a
model checkpoint.
