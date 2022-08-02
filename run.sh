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
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MAESupernetNATS --trainer_name NATSMAETrainer --log_name nats_mae_spos --work_dir ./work_dir --crit mse 

# CUDA_VISIBLE_DEVICES=3 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MAESupernetNATS --trainer_name NATSMAETrainer --log_name nats_mae_autoslim --work_dir ./work_dir --crit mse

CUDA_VISIBLE_DEVICES=6 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MAESupernetNATS --trainer_name NATSMAETrainer --log_name nats_mae_fairnas --work_dir ./work_dir --crit mse


# NATS
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name SupernetNATS --trainer_name NATSTrainer --log_name nats_spos --dataset cifar10 --log_name spos_nats --crit ce --epochs 100

# SPOS
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name SearchableShuffleNetV2 --trainer_name SPOSTrainer --log_name spos --dataset cifar10

# NB201
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name nb201 --dataset cifar10

# SPOS MAE
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name SearchableMAE --trainer_name MAETrainer --log_name spos --dataset simmim --crit mse

# MACRO
## fairnas
# CUDA_VISIBLE_DEVICES=3 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --dataset cifar10 --log_name macro_fairnas --work_dir ./work_dir --crit ce

## spos
# CUDA_VISIBLE_DEVICES=4 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --dataset cifar10 --log_name macro_spos --work_dir ./work_dir --crit ce

## spos + pairwise rank loss
# warmup : min(2, self.current_epoch/10.)
# cosine : 2 * np.sin(np.pi * 0.8 * self.current_epoch / self.max_epochs)
# CUDA_VISIBLE_DEVICES=5 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --dataset cifar10 --log_name macro_pairwise_cosine --work_dir ./work_dir --crit ce


# EVAL MACRO FLOPS
# python tools/eval_rank.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --dataset cifar10 --log_name macro_eval_flops --work_dir ./work_dir --crit ce
