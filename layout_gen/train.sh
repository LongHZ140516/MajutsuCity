#!/usr/bin/bash

# Layout Generation
python train.py \
    --sd_locked=False --accumulate_grad_batches=8 \
    --config_path=./models/cldm_v21.yaml --model_path=./models/control_v11p_sd21.ckpt \
    --learning_rate=1e-5 --batch_size=16 --image_width=512 --image_height=512 \
    --train_data_file=./datasets/seg_generation_ours_train.txt \
    --valid_data_file=./datasets/seg_generation_ours_val.txt \
    --num_gpus=4 --is_height=False \


# Height Generation
python train.py \
    --sd_locked=False --accumulate_grad_batches=8 \
    --config_path=./models/cldm_v21.yaml --model_path=./models/control_v11p_sd21.ckpt \
    --learning_rate=1e-5 --batch_size=16 --image_width=512 --image_height=512 \
    --train_data_file=./datasets/height_generation_ours_train.txt \
    --valid_data_file=./datasets/height_generation_ours_val.txt \
    --num_gpus=4 --is_height=True \