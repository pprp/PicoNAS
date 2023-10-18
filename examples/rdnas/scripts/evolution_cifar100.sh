#!/bin/bash
# evolution for run0
# CUDA_VISIBLE_DEVICES=1 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar100 \
#     --log_name evolution_spos_train_valid_c100_seed6_run0 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c100_seed6_run0/spos_train_valid_c100_seed6_run0_nb201_ckpt_0241.pth.tar


# CUDA_VISIBLE_DEVICES=1 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar100 \
#     --log_name evolution_spos_train_valid_c100_seed66_run1 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c100_seed66_run1/spos_train_valid_c100_seed66_run1_nb201_ckpt_0241.pth.tar


# CUDA_VISIBLE_DEVICES=1 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar100 \
#     --log_name evolution_spos_train_valid_c100_seed666_run2 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c100_seed666_run2/spos_train_valid_c100_seed666_run2_nb201_ckpt_0241.pth.tar

# train-test for cifar100

# seed 6 run0
CUDA_VISIBLE_DEVICES=1 python ../../exps/searcher/search_evo_nb201.py \
     \
    --data_dir ../../data/cifar --dataset cifar100 \
    --log_name evolution_spos_train_valid_c100_seed6_run0 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_test_c100_seed6_run0/spos_train_test_c100_seed6_run0_nb201_ckpt_0241.pth.tar

# seed 66 run1
CUDA_VISIBLE_DEVICES=1 python ../../exps/searcher/search_evo_nb201.py \
     \
    --data_dir ../../data/cifar --dataset cifar100 \
    --log_name evolution_spos_train_valid_c100_seed66_run1 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_test_c100_seed66_run1/spos_train_test_c100_seed66_run1_nb201_ckpt_0241.pth.tar

# seed 666 run2
CUDA_VISIBLE_DEVICES=1 python ../../exps/searcher/search_evo_nb201.py \
     \
    --data_dir ../../data/cifar --dataset cifar100 \
    --log_name evolution_spos_train_valid_c100_seed666_run2 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_test_c100_seed666_run2/spos_train_test_c100_seed666_run2_nb201_ckpt_0241.pth.tar
