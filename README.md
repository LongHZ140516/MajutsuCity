<p align="center">
  <img src="assets/logo.png" width="30%">
</p>

# MajutsuCity: Language-driven Aesthetic-adaptive City Generation with Controllable 3D Assets and Layouts

<p align="center">
  <a href="https://arxiv.org/abs/2511.20415" target="_blank">
    <img src="https://img.shields.io/badge/-arXiv-%23b91c1c?style=flat&logo=arxiv&logoColor=white&labelColor=%23b91c1c" alt="arXiv Paper">
  </a>
  <a href="https://github.com/LongHZ140516/MajutsuCity" target="_blank">
    <img src="https://img.shields.io/badge/-Github-%236EB4F9?style=flat&logo=github&logoColor=white&labelColor=%236EB4F9" alt="GitHub Repo">
  </a>
  <a href="https://longhz140516.github.io/MajutsuCity/" target="_blank">
    <img src="https://img.shields.io/badge/-ProjectPage-%23FF7FAE?style=flat&logo=googlechrome&logoColor=white&labelColor=%23FF7FAE" alt="Project Page">
  </a>
  <a href="https://huggingface.co/datasets/SereinH/MajutsuDataset" target="_blank">
    <img src="https://img.shields.io/badge/-Dataset-FFD21E?style=flat&logo=huggingface&logoColor=white&labelColor=FFD21E" alt="Project Page">
  </a>
  <a href="https://huggingface.co/SereinH/MajutsuCity/tree/main" target="_blank">
    <img src="https://img.shields.io/badge/-Model-FFD21E?style=flat&logo=huggingface&logoColor=white&labelColor=FFD21E" alt="Project Page">
  </a>
</p>

<p align="center">
  <img src="assets/flag.jpg" width="95%">
</p>


## 🏗️ Pipeline Overview

<p align="center">
  <img src="assets/pipeline.jpg" width="95%">
</p>

## 📜 TODO List
- [x] Release ${\color{#8D88E2}MajutsuDataset}$
- [ ] Release ${\color{#6EB4F9}MajutsuCity}$
  - [x] Layout Generation model
  - [x] Material texture fintuned model
  - [ ] Framework code
- [ ] Release ${\color{#FF7FAE}MajutsuAgent}$ code

## 🪄 Usage

### Installation

```bash
git clone https://github.com/LongHZ140516/MajutsuCity.git
cd MajutsuCity

conda env create -f environment.yml
conda activate majutsucity
```

### Layout Generation

Download the [*layout ckpt model*](https://huggingface.co/SereinH/MajutsuCity/tree/main/layout) to the folder `ckpt/layout`.

(If you want to train the model, please refer to the instructions of [Layout Training README](./layout_gen/README.md).)

```bash
cd layout_gen
# Layout Case
python xxx.py
```

### Texture / Skybox Generation

Download the [*texture ckpt model*](https://huggingface.co/SereinH/MajutsuCity/tree/main/layout) to the folder `ckpt/texture`

```bash
# Texture Case
python texture_gen.py

# Skybox Case
python skybox_gen.py
```

## 🙏 Acknowledgements

- Some visual design inspirations (e.g., icons and layout ideas) are adapted from [awesome-framework-gallery](https://github.com/LongHZ140516/awesome-framework-gallery), [Arknights](https://ak.hypergryph.com/#index) and [Arknights: Endfield](https://endfield.hypergryph.com/).
- The *Texture/Skybox LoRA* is trained on the basis of [flymyai-lora-trainer](https://github.com/FlyMyAI/flymyai-lora-trainer). 

## 📝 Citation

```bib
@article{huang2025majutsucity,
  title={MajutsuCity: Language-driven Aesthetic-adaptive City Generation with Controllable 3D Assets and Layouts},
  author={Huang, Zilong and He, Jun and Huang, Xiaobin and Xiong, Ziyi and Luo, Yang and Ye, Junyan and Li, Weijia and Chen, Yiping and Han, Ting},
  journal={arXiv preprint arXiv:2511.20415},
  year={2025}
}
```


