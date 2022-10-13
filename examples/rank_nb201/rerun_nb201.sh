#!/usr/bin/env sh

NUM_GPUS=1

MKL_NUM_THREADS=4
OMP_NUM_THREADS=1

# Note that: BN affine is set to False

# 1. spos with uniform sampling strategy
CUDA_VISIBLE_DEVICES=0 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name normal_nb201_spos_uniform-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type uniform --data_dir ../../data/cifar

# evolution search
CUDA_VISIBLE_DEVICES=2 python ../../tools/searcher/search_evo_nb201.py --config ../../configs/spos/spos_cifar10.py --model_path '/home/stack/project/spos-cifar/examples/rank_nb201/checkpoints/normal_nb201_spos_uniform-sampling_exp-rerun/normal_nb201_spos_uniform-sampling_exp-rerun_nb201_ckpt_0191.pth.tar' --data_dir ../../data/cifar


# # 2. FairNAS
CUDA_VISIBLE_DEVICES=1 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name normal_nb201_fairnas_fairsampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type fair --data_dir ../../data/cifar

# 3. Sandwich Rule
CUDA_VISIBLE_DEVICES=2 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name normal_nb201_spos_uniform-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type sandwich  --data_dir ../../data/cifar

# 4. Random Sample + Pairwise Ranking
CUDA_VISIBLE_DEVICES=0 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name pairwise-ranking_nb201_spos_random-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type random --data_dir ../../data/cifar

# 5. Hamming Sample + Pairwise Ranking
CUDA_VISIBLE_DEVICES=0 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name pairwise-ranking_nb201_spos_hamming-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type hamming --data_dir ../../data/cifar

# 6. Adaptive Sample + Pairwise Ranking
CUDA_VISIBLE_DEVICES=1 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name pairwise-ranking_nb201_spos_adaptive-sampling_exp-rerun --dataset cifar10 --crit ce --lr 0.025 --type adaptive --data_dir ../../data/cifar

#######################################################################################
# Note: run with zero operation.
# check the oneshot_nasbench201
CUDA_VISIBLE_DEVICES=0 python ../../tools/train.py --config ../../configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name normal_nb201_spos_uniform-sampling_exp_with_zero --dataset cifar10 --crit ce --lr 0.025 --type uniform --data_dir ../../data/cifar
