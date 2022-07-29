#!/usr/bin/env sh

NUM_GPUS=1

MKL_NUM_THREADS=4
OMP_NUM_THREADS=1

# srun --partition=mm_model \
#     --job-name=autoformer_reimplement \
#     --gres=gpu:$NUM_GPUS \
#     --ntasks=$NUM_GPUS \
#     --ntasks-per-node=$NUM_GPUS \
#     --cpus-per-task=$NUM_GPUS \
#     --kill-on-bad-exit=1 \
#     --quotatype=auto \
#     --async \
#     python

# NATS MAE
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name 'MAESupernetNATS' --trainer_name 'NATSMAETrainer'

# NATS
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name 'SupernetNATS' --trainer_name 'NATSTrainer' --log_name "nats_spos" --dataset "cifar10"
