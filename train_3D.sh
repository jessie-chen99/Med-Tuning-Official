#!/bin/bash
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch \
--nproc_per_node 8 \
--master_port 12395 \
train_3D.py \
--batch_size 2 \
--lr 2e-3 \
--end_epoch 250 \
--warmup_epoch 25 \
--dataset_name BraTS2019 \
--root ./Datasets/BraTS2019/ \
--preckpt supervised \
--transfer_type med_adapter \
--backbone_type ViTB16 \
--experiment exp_name
