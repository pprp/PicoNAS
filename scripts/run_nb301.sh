#!/bin/bash

# 命名规范：{search_space}_{method}_{description}_{exp NO.}
# 记录内容：运行时间，图表下载 (tensorboard)
# 注意事项：备份结果, 检查代码和执行bash是否一致。

CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench301Network --trainer_name NB301Trainer --log_name nb301_spos_baseline_exp1.0 --dataset cifar10 --crit ce --lr 0.025 --type uniform
