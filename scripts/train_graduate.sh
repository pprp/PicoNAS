#!/bin/bash

# 命名规范：graduate_{search_space}_{method}_{description}_{exp NO.}
# 记录内容：运行时间，图表下载 (tensorboard)
# 注意事项：备份结果。


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";
start=`date +%s`

# 第四章 实验1.0 max subnet
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_maxsubnet_exp1.0 --dataset cifar10 --crit ce --lr 0.025

# 第四章 实验1.1 mid subnet
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_midsubnet_exp1.1 --dataset cifar10 --crit ce --lr 0.025

# 第四章 实验1.2 min subnet
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_minsubnet_exp1.2 --dataset cifar10 --crit ce --lr 0.025

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
