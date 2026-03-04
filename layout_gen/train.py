import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from share import *
from dataset import LayoutDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def create_argparser():
    defaults = dict(
        seed = 42,
        # model
        result_dir = './results',
        config_path = './models/cldm_v15.yaml',
        model_path = './models/control_sd15_ini.ckpt',
        sd_locked = True,
        only_mid_control = False,
        learning_rate = 1e-5,
        # data
        batch_size = 4,
        image_width = 512,
        image_height = 256,
        drop_prompt_ratio = 0.0,
        text_prompt = '',
        train_data_file = '',
        valid_data_file = '',
        # train
        max_epochs = 100,
        logger_freq = 3000,
        num_gpus = 3,
        accumulate_grad_batches = 1,
        unconditional_guidance_scale = 9.0,
        is_cvusa = False,
        negative_prompt = ""
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def main():
    args = create_argparser().parse_args()

    now = datetime.now().strftime("%Y%m%d%H%M")
    result_dir = os.path.abspath(os.path.join(args.result_dir, now))
    makedir(result_dir)
    (Path(result_dir) / 'args.json').write_text(json.dumps(args.__dict__, indent=4))

    pl.seed_everything(args.seed, workers=True)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config_path).cpu()
    model.load_state_dict(load_state_dict(args.model_path, location='cpu'), strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    # Misc
    image_size = (args.image_width, args.image_height)
    train_dataset = LayoutDataset(args.train_data_file, args.is_cvusa, args.text_prompt, image_size, args.drop_prompt_ratio)
    valid_dataset = LayoutDataset(args.valid_data_file, args.is_cvusa, args.text_prompt, image_size)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)

    log_images_kwargs = {
        'unconditional_guidance_scale': args.unconditional_guidance_scale,
    }
    logger = ImageLogger(batch_frequency=args.logger_freq, log_images_kwargs=log_images_kwargs)
    checkpoint_callback = ModelCheckpoint(dirpath=result_dir, monitor='val/loss_simple')
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy="ddp",
        precision=32,
        max_epochs=args.max_epochs,
        default_root_dir=result_dir,
        callbacks=[logger, checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batches
    )


   

    # Train!
    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()