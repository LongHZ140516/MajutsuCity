import os
import torch
import argparse
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from share import *
from dataset import LayoutDataset
from cldm.logger import ValidImageLogger
from cldm.model import create_model, load_state_dict


def create_argparser():
    defaults = dict(
        seed = 2025,
        config_path = './models/cldm_v15.yaml',
        model_path = './models/control_sd15_ini.ckpt',
        image_width = 512,
        image_height = 256,
        text_prompt = '',
        data_file_path = '',
        batch_size = 4,
        result_dir = './results',
        logger_freq = 1,
        sample_num = 10000,
        unconditional_guidance_scale = 9.0,
        num_gpus = 8,
        is_height = False,
        negative_prompt = ""
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    model = create_model(args.config_path).cpu()
    # Release any CUDA cache left by sub-module __init__ calls (e.g. longclip.load)
    # before DDP assigns each process its own GPU.
    torch.cuda.empty_cache()

    state_dict = load_state_dict(args.model_path, location='cpu')
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Exclude the frozen LongCLIP backbone (already loaded from the .pt file by
    # create_model), but DO load the trained text_projection so text conditioning
    # actually works at inference time.
    filtered_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("cond_stage_model.")
        or k.startswith("cond_stage_model.text_projection")
    }
    model.load_state_dict(filtered_dict, strict=False)
    # Do NOT call model.cuda() here — with DDP each process has its own GPU and
    # Lightning's model_to_device() will move the model to the correct device.
    # Calling .cuda() before DDP setup sends everything to GPU 0 across all 8
    # processes simultaneously, causing OOM.
    model.eval()

    image_size = (args.image_width, args.image_height)
    print(args.data_file_path)
    print(args.is_height)
    print('neg_prompt', args.negative_prompt)
    print('args.resultdir:',args.result_dir)
    dataset = LayoutDataset(args.data_file_path, args.is_height, args.text_prompt, image_size)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)

    log_images_kwargs = {
        'unconditional_guidance_scale': args.unconditional_guidance_scale,
        'negative_prompt': args.negative_prompt
    }
    logger = ValidImageLogger(args.result_dir, args.logger_freq, log_images_kwargs=log_images_kwargs)
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy="ddp",
        precision=32,
        max_epochs=1,
        callbacks=[logger]
    )
    trainer.validate(model, dataloader)

if __name__ == '__main__':
    main()
