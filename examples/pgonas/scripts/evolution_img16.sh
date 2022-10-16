#!/bin/bash
# evolution for run0
CUDA_VISIBLE_DEVICES=2 python ../../tools/searcher/search_evo_nb201.py \
    --config ../../configs/spos/spos_cifar10.py \
    --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
    --log_name evolution_spos_train_valid_img16_seed6_run0 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_img16_seed6_run0/spos_train_valid_img16_seed6_run0_nb201_ckpt_0241.pth.tar


CUDA_VISIBLE_DEVICES=2 python ../../tools/searcher/search_evo_nb201.py \
    --config ../../configs/spos/spos_cifar10.py \
    --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
    --log_name evolution_spos_train_valid_img16_seed66_run1 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_img16_seed66_run1/spos_train_valid_img16_seed66_run1_nb201_ckpt_0241.pth.tar


CUDA_VISIBLE_DEVICES=2 python ../../tools/searcher/search_evo_nb201.py \
    --config ../../configs/spos/spos_cifar10.py \
    --data_dir ../../data/ImageNet16 --dataset ImageNet16-120 \
    --log_name evolution_spos_train_valid_img16_seed666_run2 \
    --model_path /home/stack/project/spos-cifar/examples/pgonas/checkpoints/spos_train_valid_img16_seed666_run2/spos_train_valid_img16_seed666_run2_nb201_ckpt_0241.pth.tar
