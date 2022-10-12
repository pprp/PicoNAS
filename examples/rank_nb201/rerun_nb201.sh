#!/usr/bin/env sh

NUM_GPUS=1

MKL_NUM_THREADS=4
OMP_NUM_THREADS=1

# Note that: BN affine is set to False

# 1. spos with uniform sampling strategy
CUDA_VISIBLE_DEVICES=0 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name normal_nb201_spos_uniform-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type uniform --data_dir ../../data/cifar

# # 2. FairNAS
CUDA_VISIBLE_DEVICES=1 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name normal_nb201_fairnas_fairsampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type fair --data_dir ../../data/cifar

# 3. Sandwich Rule
CUDA_VISIBLE_DEVICES=2 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name normal_nb201_spos_uniform-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type sandwich  --data_dir ../../data/cifar

# 4. Random Sample + Pairwise Ranking
CUDA_VISIBLE_DEVICES=2 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name pairwise-ranking_nb201_spos_uniform-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type random --data_dir ../../data/cifar

# 5. Hamming Sample + Pairwise Ranking
CUDA_VISIBLE_DEVICES=0 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name pairwise-ranking_nb201_spos_uniform-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type hamming --data_dir ../../data/cifar

# 6. Adaptive Sample + Pairwise Ranking
CUDA_VISIBLE_DEVICES=1 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name pairwise-ranking_nb201_spos_uniform-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type adaptive --data_dir ../../data/cifar
