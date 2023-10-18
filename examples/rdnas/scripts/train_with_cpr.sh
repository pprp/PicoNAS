#!/bin/bash

# SPOS
# CUDA_VISIBLE_DEVICES=1 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --log_name evaluate_cpr_spos \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --crit ce --lr 0.025 --type uniform


# # FairNAS
# CUDA_VISIBLE_DEVICES=1 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --log_name evaluate_cpr_fairnas \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --crit ce --lr 0.025 --type fair


# PGONAS
CUDA_VISIBLE_DEVICES=1 python ../../exps/train.py \
     \
    --model_name OneShotNASBench201Network \
    --trainer_name NB201Trainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/cifar --dataset cifar10 \
    --type adaptive  \
    --seed 6 \
    --log_name evaluate_cpr_pgonas
