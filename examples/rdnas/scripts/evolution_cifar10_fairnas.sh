#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
     \
    --data_dir ../../data/cifar --dataset cifar10 \
    --log_name evolution_spos_train_valid_c10_seed666_run2_rerun \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c10_seed666_run2/spos_train_valid_c10_seed666_run2_nb201_ckpt_0241.pth.tar
