#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./jahs_ranker.py --config ../../configs/spos/spos_cifar10.py --data_dir ../../data/cifar --ckpt_path /home/stack/project/spos-cifar/checkpoints/graduate_nb201_spos_uniform-sampling_exp3.0/graduate_nb201_spos_uniform-sampling_exp3.0_nb201_ckpt_0151.pth.tar
