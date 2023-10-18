#!/bin/bash

# ------------------------ train pgonas with train-valid
CUDA_VISIBLE_DEVICES=0 python train_rdnas.py \
     \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/cifar --dataset cifar10 \
    --type adaptive  \
    --seed 6 \
    --log_name spos_train_valid_c10_seed6_run0

# CUDA_VISIBLE_DEVICES=0 python train_rdnas.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 66 \
#     --log_name spos_train_valid_c10_seed66_run1

# CUDA_VISIBLE_DEVICES=0 python train_rdnas.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 666 \
#     --log_name spos_train_valid_c10_seed666_run2

# ------------------------- train pgo-nas with train-test
# CUDA_VISIBLE_DEVICES=0 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 6 \
#     --log_name spos_train_test_c10_seed6_run0

# CUDA_VISIBLE_DEVICES=0 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 66 \
#     --log_name spos_train_test_c10_seed66_run1

# CUDA_VISIBLE_DEVICES=0 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 666 \
#     --log_name spos_train_test_c10_seed666_run2

# CUDA_VISIBLE_DEVICES=1 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 42 \
#     --log_name spos_train_test_c10_seed42_run3

# CUDA_VISIBLE_DEVICES=1 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 43 \
#     --log_name spos_train_test_c10_seed43_run4
