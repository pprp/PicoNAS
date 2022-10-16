#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python train_pgonas.py \
#     --config ../../configs/spos/spos_cifar10.py \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 6 \
#     --log_name spos_train_valid_c10_seed6_run0

# CUDA_VISIBLE_DEVICES=0 python train_pgonas.py \
#     --config ../../configs/spos/spos_cifar10.py \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 66 \
#     --log_name spos_train_valid_c10_seed66_run1

# CUDA_VISIBLE_DEVICES=0 python train_pgonas.py \
#     --config ../../configs/spos/spos_cifar10.py \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 666 \
#     --log_name spos_train_valid_c10_seed666_run2

CUDA_VISIBLE_DEVICES=0 python train_pgonas.py \
    --config ../../configs/spos/spos_cifar10.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/cifar --dataset cifar10 \
    --type adaptive  \
    --seed 6666 \
    --log_name spos_train_valid_c10_seed6666_run3

CUDA_VISIBLE_DEVICES=0 python train_pgonas.py \
    --config ../../configs/spos/spos_cifar10.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/cifar --dataset cifar10 \
    --type adaptive  \
    --seed 66666 \
    --log_name spos_train_valid_c10_seed66666_run4

CUDA_VISIBLE_DEVICES=0 python train_pgonas.py \
    --config ../../configs/spos/spos_cifar10.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/cifar --dataset cifar10 \
    --type adaptive  \
    --seed 666666 \
    --log_name spos_train_valid_c10_seed666666_run5
