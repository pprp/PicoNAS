#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python train_pgonas.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
#     --type adaptive  \
#     --seed 6 \
#     --log_name spos_train_valid_img16_seed6_run0


# CUDA_VISIBLE_DEVICES=2 python train_pgonas.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
#     --type adaptive  \
#     --seed 66 \
#     --log_name spos_train_valid_img16_seed66_run1


# CUDA_VISIBLE_DEVICES=2 python train_pgonas.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
#     --type adaptive  \
#     --seed 666 \
#     --log_name spos_train_valid_img16_seed666_run2


# ------------------------- train pgo-nas with train-test

CUDA_VISIBLE_DEVICES=2 python ../../exps/train.py \
     \
    --model_name OneShotNASBench201Network \
    --trainer_name NB201Trainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
    --type adaptive  \
    --seed 6 \
    --log_name spos_train_test_img16_seed6_run0


CUDA_VISIBLE_DEVICES=2 python ../../exps/train.py \
     \
    --model_name OneShotNASBench201Network \
    --trainer_name NB201Trainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
    --type adaptive  \
    --seed 66 \
    --log_name spos_train_test_img16_seed66_run1


CUDA_VISIBLE_DEVICES=2 python ../../exps/train.py \
     \
    --model_name OneShotNASBench201Network \
    --trainer_name NB201Trainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 64 \
    --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
    --type adaptive  \
    --seed 666 \
    --log_name spos_train_test_img16_seed666_run2
