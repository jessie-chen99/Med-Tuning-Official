#!/bin/bash
CUDA_VISIBLE_DEVICES='0' python test_offi_3D.py \
--experiment exp_name \
--backbone_type ViTB16 \
--root ./Datasets/BraTS2019/ \
--preckpt supervised \
--transfer_type full_finetuning_3D 
