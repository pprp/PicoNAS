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
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MAESupernetNATS --trainer_name NATSMAETrainer --log_name baseline --work_dir ./work_dir

# NATS
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name SupernetNATS --trainer_name NATSTrainer --log_name nats_spos --dataset cifar10 --log_name spos_nats --crit ce --epochs 100

# SPOS
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name SearchableShuffleNetV2 --trainer_name SPOSTrainer --log_name spos --dataset cifar10

# NB201
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name nb201 --dataset cifar10

# SPOS MAE
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name SearchableMAE --trainer_name MAETrainer --log_name spos --dataset simmim --crit mse

# MACRO
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name macro --dataset cifar10 --log_name macro --work_dir ./work_dir --crit ce
