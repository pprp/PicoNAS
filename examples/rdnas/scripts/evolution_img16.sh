#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
#     --log_name evolution_spos_train_valid_img16_seed6_run0 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_img16_seed6_run0/spos_train_valid_img16_seed6_run0_nb201_ckpt_0241.pth.tar


# CUDA_VISIBLE_DEVICES=2 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
#     --log_name evolution_spos_train_valid_img16_seed66_run1 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_img16_seed66_run1/spos_train_valid_img16_seed66_run1_nb201_ckpt_0241.pth.tar


# CUDA_VISIBLE_DEVICES=2 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
#     --log_name evolution_spos_train_valid_img16_seed666_run2 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_img16_seed666_run2/spos_train_valid_img16_seed666_run2_nb201_ckpt_0241.pth.tar

# CUDA_VISIBLE_DEVICES=2 python ../../exps/searcher/search_evo_nb201.py \
#      --seed 20221017 \
#     --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
#     --log_name evolution_spos_train_valid_img16_seed666_run3 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_img16_seed666_run2/spos_train_valid_img16_seed666_run2_nb201_ckpt_0241.pth.tar

# CUDA_VISIBLE_DEVICES=2 python ../../exps/searcher/search_evo_nb201.py \
#      --seed 42 \
#     --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
#     --log_name evolution_spos_train_valid_img16_seed666_run4 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_img16_seed666_run2/spos_train_valid_img16_seed666_run2_nb201_ckpt_0241.pth.tar

# train-test for img16

CUDA_VISIBLE_DEVICES=1 python ../../exps/searcher/search_evo_nb201.py \
     --seed 42 \
    --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
    --log_name evolution_spos_train_valid_img16_seed6_run0 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_test_img16_seed6_run0/spos_train_test_img16_seed6_run0_nb201_ckpt_0241.pth.tar
