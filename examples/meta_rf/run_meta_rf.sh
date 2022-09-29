#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python ./train_imagenet.py --config ../../configs/spos/spos_cifar10.py --model_name MobileNetv2MetaReceptionField --trainer_name MetaTrainer  --dataset cifar10 --crit ce --lr 0.05  --log_name meta_train_mbv2_rf --batch_size 64 --epochs 100 --sched cosine  --data_dir ../../data/cifar
