#!/bin/bash

# 命名规范：graduate_{search_space}_{method}_{description}_{exp NO.}
# 记录内容：运行时间，图表下载 (tensorboard)
# 注意事项：备份结果, 检查代码和执行bash是否一致。


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";
start=`date +%s`

# 第四章 实验1.0 max subnet
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_maxsubnet_exp1.0 --dataset cifar10 --crit ce --lr 0.025

# 第四章 实验1.1 mid subnet
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_midsubnet_exp1.1 --dataset cifar10 --crit ce --lr 0.025

# 第四章 实验1.2 min subnet
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_minsubnet_exp1.2 --dataset cifar10 --crit ce --lr 0.025

# 第四章 实验1.3 alter subnet
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Alter_Trainer --log_name graduate_nb201_spos_altersubnet_exp1.3 --dataset cifar10 --crit ce --lr 0.025

# 第四章 实验3.0 uniform sampling cifar10
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_uniformsampling_exp3.0 --dataset cifar10 --crit ce --lr 0.025

# 第四章 实验3.1 fair sampling cifar10
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_fairnas_fairsampling_exp3.1 --dataset cifar10 --crit ce --lr 0.025


# 第五章 实验2.0 macro benchmark + random type
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_randomtype_exp2.0 --dataset cifar10 --crit ce --lr 0.025

# 第五章 实验2.1 macro benchmark + hamming type
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_hammingtype_exp2.1 --dataset cifar10 --crit ce --lr 0.025

# # 第五章 实验2.2 macro benchmark + adaptive hamming type
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_adaptivetype_exp2.2 --dataset cifar10 --crit ce --lr 0.025


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
