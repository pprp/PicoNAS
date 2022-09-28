#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../tools/train_shrinker.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer  --dataset cifar10 --crit ce --lr 0.05 --type random  --log_name anaylse_nb201_pairwise-random_exp1.3_align --expand_times 0 --shrink_times 0 --batch_size 512 --epochs 1000 --sched plateau
