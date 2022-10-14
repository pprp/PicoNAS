#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_pgonas.py \
    --config ../../configs/spos/spos_cifar10.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/cifar --dataset cifar100 \
    --type adaptive  \
    --log_name spos_train_valid_c100_run0
