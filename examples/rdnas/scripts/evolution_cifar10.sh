#!/bin/bash
# evolution for run0
# CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --log_name evolution_spos_train_valid_c10_seed6_run0 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c10_seed6_run0/spos_train_valid_c10_seed6_run0_nb201_ckpt_0241.pth.tar

# CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --log_name evolution_spos_train_valid_c10_seed66_run1 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c10_seed66_run1/spos_train_valid_c10_seed66_run1_nb201_ckpt_0241.pth.tar

CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
     \
    --data_dir ../../data/cifar --dataset cifar10 \
    --log_name evolution_spos_train_valid_c10_seed666_run2_rerun \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c10_seed666_run2/spos_train_valid_c10_seed666_run2_nb201_ckpt_0241.pth.tar

CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
     \
    --data_dir ../../data/cifar --dataset cifar10 \
    --log_name evolution_spos_train_valid_c10_seed666_run3_rerun \
    --seed 42 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c10_seed666_run2/spos_train_valid_c10_seed666_run2_nb201_ckpt_0241.pth.tar


CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
     \
    --data_dir ../../data/cifar --dataset cifar10 \
    --log_name evolution_spos_train_valid_c10_seed666_run4_rerun \
    --seed 42 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_c10_seed666_run2/spos_train_valid_c10_seed666_run2_nb201_ckpt_0231.pth.tar

# # search for train-test of cifar10
# CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --log_name evolution_spos_train_test_c10_seed6_run0_fixed \
#     --seed 42 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_test_c10_seed6_run0/spos_train_test_c10_seed6_run0_nb201_ckpt_0241.pth.tar

# # seed 66 run 1
# CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --log_name evolution_spos_train_test_c10_seed66_run1_fixed \
#     --seed 42 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_test_c10_seed66_run1/spos_train_test_c10_seed66_run1_nb201_ckpt_0241.pth.tar

# # seed 42 run 3
# CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --log_name evolution_spos_train_test_c10_seed42_run3_fixed \
#     --seed 42 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_test_c10_seed42_run3/spos_train_test_c10_seed42_run3_nb201_ckpt_0241.pth.tar

# # seed 43 run 4
# CUDA_VISIBLE_DEVICES=0 python ../../exps/searcher/search_evo_nb201.py \
#      \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --log_name evolution_spos_train_test_c10_seed43_run4_fixed \
#     --seed 42 \
#     --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_test_c10_seed43_run4/spos_train_test_c10_seed43_run4_nb201_ckpt_0241.pth.tar
