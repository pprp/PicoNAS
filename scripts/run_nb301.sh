#!/bin/bash

# 命名规范：{search_space}_{method}_{description}_{x NO.}
# 记录内容：运行时间，图表下载 (tensorboard)
# 注意事项：备份结果, 检查代码和执行bash是否一致。

CUDA_VISIBLE_DEVICES= python exps/train.py  --model_name OneShotNASBench301Network --trainer_name NB301Trainer --log_name nb301_spos_baseline_x1 --dataset cifar10 --crit ce --lr 0.025 --type uniform

CUDA_VISIBLE_DEVICES=0 python exps/train.py  --model_name OneShotNASBench301Network --trainer_name NB301Trainer --log_name nb301_spos_random_x1 --dataset cifar10 --crit ce --lr 0.025 --type random

CUDA_VISIBLE_DEVICES=0 python exps/train.py  --model_name OneShotNASBench301Network --trainer_name NB301Trainer --log_name nb301_spos_hamming_x1 --dataset cifar10 --crit ce --lr 0.025 --type hamming



CUDA_VISIBLE_DEVICES=1 python exps/train.py  --model_name OneShotNASBench301Network --trainer_name NB301Trainer --log_name nb301_spos_adaptive_x1 --dataset cifar10 --crit ce --lr 0.025 --type adaptive


# CUDA_VISIBLE_DEVICES=2 python exps/train.py  --model_name OneShotNASBench301Network --trainer_name NB301Trainer --log_name nb301_spos_fair_x1 --dataset cifar10 --crit ce --lr 0.025 --type fair
