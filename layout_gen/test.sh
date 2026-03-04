#!/usr/bin/bash
python test.py \
    --num_gpus=4 \
    --config_path=./models/cldm_v21.yaml \
    --image_width=512 --image_height=512 \
    --result_dir=./Output_Image_Dir \
    --model_path=./results/202511132024/epoch=45-step=919.ckpt \
    --data_file_path=./datasets/seg_generation_ours_val.txt \
    --seed 42
