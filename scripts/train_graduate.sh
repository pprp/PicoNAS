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

#########################################################################################################################################################################

# 第四章 实验2.1 基于flops采样
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_flopspolicysampler_exp2.1 --dataset cifar10 --crit ce --lr 0.025
# 进化采样
# CUDA_VISIBLE_DEVICES=1 python tools/searcher/search_evo_nb201.py --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_flopspolicysampler_exp2.1/graduate_nb201_balanced_flopspolicysampler_exp2.1_nb201_ckpt_0191.pth.tar'

# 第四章 实验2.1_c100 基于flops采样
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_flops-policy-sampler-cifar100_exp2.1_c100 --dataset cifar100 --crit ce --lr 0.1 --type flops
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_flops-policy-sampler-cifar100_exp2.1_c100 --dataset cifar100 --crit ce  --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_flops-policy-sampler-cifar100_exp2.1_c100/graduate_nb201_balanced_flops-policy-sampler-cifar100_exp2.1_c100_nb201_ckpt_0191.pth.tar'

# 第四章 实验2.1_img16 基于flops采样
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_flops-policy-sampler-imagenet16_exp2.1_img16 --dataset imagenet16 --crit ce --lr 0.1 --type flops --data_dir ./data/ImageNet16
# CUDA_VISIBLE_DEVICES=2 python tools/searcher/search_evo_nb201.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_flops-policy-sampler-imagenet16_exp2.1_img16 --dataset imagenet16 --crit ce --lr 0.1 --data_dir ./data/ImageNet16  --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_flops-policy-sampler-imagenet16_exp2.1_img16/graduate_nb201_balanced_flops-policy-sampler-imagenet16_exp2.1_img16_nb201_ckpt_0171.pth.tar'


# 第四章 实验2.2 基于params采样
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_paramspolicysampler_exp2.2 --dataset cifar10 --crit ce --lr 0.025
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_paramspolicysampler_exp2.2/graduate_nb201_balanced_paramspolicysampler_exp2.2_nb201_ckpt_0191.pth.tar'


# 第四章 实验2.2_c100 基于params采样
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_params-policy-sampler-cifar100_exp2.2_c100 --dataset cifar100 --crit ce --lr 0.1 --type params
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_graduate_nb201_balanced_params-policy-sampler-cifar100_exp2.2_c100 --dataset cifar100 --crit ce  --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_params-policy-sampler-cifar100_exp2.2_c100/graduate_nb201_balanced_params-policy-sampler-cifar100_exp2.2_c100_nb201_ckpt_0171.pth.tar'

# 第四章 实验2.2_img16 基于params采样
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_params-policy-sampler-imagenet16_exp2.2_img16 --dataset imagenet16 --crit ce --lr 0.1 --type params --data_dir ./data/ImageNet16
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_graduate_nb201_balanced_params-policy-sampler-imagenet16_exp2.2_img16 --dataset imagenet16 --crit ce --lr 0.1 --data_dir ./data/ImageNet16  --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_params-policy-sampler-imagenet16_exp2.2_img16/graduate_nb201_balanced_params-policy-sampler-imagenet16_exp2.2_img16_nb201_ckpt_0181.pth.tar'


# 第四章 实验2.3 基于nwot采样
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_nwot-policysampler_exp2.3 --dataset cifar10 --crit ce --lr 0.025
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_nwot-policysampler_exp2.3/graduate_nb201_balanced_nwot-policysampler_exp2.3_nb201_ckpt_0171.pth.tar'


# 第四章 实验2.3_c100 基于nwot采样
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_nwot-policy-sampler-cifar100_exp2.3_c100 --dataset cifar100 --crit ce --lr 0.1 --type nwot
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_graduate_nb201_balanced_nwot-policy-sampler-cifar100_exp2.3_c100 --dataset cifar100 --crit ce  --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_nwot-policy-sampler-cifar100_exp2.3_c100/graduate_nb201_balanced_nwot-policy-sampler-cifar100_exp2.3_c100_nb201_ckpt_0171.pth.tar'

# 第四章 实验2.3_img16 基于nwot采样
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_nwot-policy-sampler-imagenet16_exp2.3_img16 --dataset imagenet16 --crit ce --lr 0.1 --type nwot --data_dir ./data/ImageNet16
# CUDA_VISIBLE_DEVICES=2 python tools/searcher/search_evo_nb201.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_graduate_nb201_balanced_nwot-policy-sampler-imagenet16_exp2.3_img16 --dataset imagenet16 --crit ce --lr 0.1 --data_dir ./data/ImageNet16  --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_nwot-policy-sampler-imagenet16_exp2.3_img16/graduate_nb201_balanced_nwot-policy-sampler-imagenet16_exp2.3_img16_nb201_ckpt_0161.pth.tar'



# 第四章 实验2.4 基于zenscore采样
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_zenscore-policysampler_exp2.4 --dataset cifar10 --crit ce --lr 0.025
# CUDA_VISIBLE_DEVICES=1 python tools/searcher/search_evo_nb201.py --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_zenscore-policysampler_exp2.4/graduate_nb201_balanced_zenscore-policysampler_exp2.4_nb201_ckpt_0181.pth.tar'

# 第四章 实验2.4_c100 基于zenscore采样
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_zenscore-policy-sampler-cifar100_exp2.4_c100 --dataset cifar100 --crit ce --lr 0.1 --type zenscore
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_graduate_nb201_balanced_zenscore-policy-sampler-cifar100_exp2.4_c100 --dataset cifar100 --crit ce  --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_zenscore-policy-sampler-cifar100_exp2.4_c100/graduate_nb201_balanced_zenscore-policy-sampler-cifar100_exp2.4_c100_nb201_ckpt_0191.pth.tar'

# 第四章 实验2.4_img16 基于zenscore采样
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_balanced_zenscore-policy-sampler-imagenet16_exp2.4_img16 --dataset imagenet16 --crit ce --lr 0.1 --type zenscore --data_dir ./data/ImageNet16
# CUDA_VISIBLE_DEVICES=2 python tools/searcher/search_evo_nb201.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_graduate_nb201_balanced_zenscore-policy-sampler-imagenet16_exp2.4_img16 --dataset imagenet16 --crit ce --lr 0.1 --data_dir ./data/ImageNet16  --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_balanced_zenscore-policy-sampler-imagenet16_exp2.4_img16/graduate_nb201_balanced_zenscore-policy-sampler-imagenet16_exp2.4_img16_nb201_ckpt_0191.pth.tar'


#########################################################################################################################################################################

# 第四章 实验3.0 uniform sampling cifar10
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_uniform-sampling_exp3.0 --dataset cifar10 --crit ce --lr 0.025
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_spos_uniform-sampling_exp3.0/graduate_nb201_spos_uniform-sampling_exp3.0_nb201_ckpt_0151.pth.tar'

# TODO search space shrinking .....
CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrink_nb201_spos_uniform-sampling_exp3.0 --dataset cifar10 --crit ce --lr 0.025 --type random

# TODO search space shrink and expand ...
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0 --dataset cifar10 --crit ce --lr 0.025 --type random

# TODO search space shrink and expand ... [fixbug]
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_fixbug --dataset cifar10 --crit ce --lr 0.025 --type random

# TODO search space shrink and expand ... [adjust zero shot metric]
# TODO origin is nwot; current is flops
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO sample num set to 200, origin is 12
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO shrink 2 times
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_shrink_2_times --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO only expand
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_only-expand --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO expand every 20 epochs
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_expand_every_20_epochs --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO expand every 5 epochs
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_expand_every_5_epochs --dataset cifar10 --crit ce --lr 0.025 --type uniform


# TODO expand more 12
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_expand-more --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO enlarge the search space
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_enlarge --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO enlarge the search space more times 12
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_enlarge_more --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO shrink 1 time and expand 2 times. Failed [find bug]
CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_s1e2 --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO retest the implimentation of expand and shrink.
# MODIFICATION: rewrite the expand and enlarge function and add noise.
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_shrink1expand2 --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO only expand once with noise
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_expand1 --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO no expand or shrink
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-reimplement --dataset cifar10 --crit ce --lr 0.025 --type uniform

# TODO expand 6 times and shrink 3 times [after expand] 
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201ShrinkTrainer --log_name ssshrinkexpand_nb201_spos_uniform-sampling_exp3.0_flops-guide-metric-200_sample-num_expand6shrink3 --dataset cifar10 --crit ce --lr 0.025 --type uniform

# 第四章 实验3.1 fair sampling cifar10
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_fairnas_fairsampling_exp3.1 --dataset cifar10 --crit ce --lr 0.025
# 进化搜索
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_fairnas_fairsampling_exp3.1/graduate_nb201_fairnas_fairsampling_exp3.1_nb201_ckpt_0191.pth.tar'

# 第四章 实验3.2 sandwich sampling cifar10
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_sandwich-sampling_exp3.2 --dataset cifar10 --crit ce --lr 0.025
# 进化搜索
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_path ''

# 第四章 实验3.0_c100 uniform sampling cifar100
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_uniform-sampling-cifar100_exp3.0_c100 --dataset cifar100 --crit ce --lr 0.025 --type uniform
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_nb201_spos_uniform-sampling-cifar100_exp3.0_c100 --dataset cifar100 --crit ce --lr 0.025 --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_spos_uniform-sampling-cifar100_exp3.0_c100/graduate_nb201_spos_uniform-sampling-cifar100_exp3.0_c100_nb201_ckpt_0191.pth.tar'

# 第四章 实验3.1_c100 fair sampling cifar100
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_fairnas_fairsampling-cifar100_exp3.1_c100 --dataset cifar100 --crit ce --lr 0.025 --type fair
# 进化搜索
# CUDA_VISIBLE_DEVICES=1 python tools/searcher/search_evo_nb201.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_nb201_fairnas_fairsampling-cifar100_exp3.1_c100 --dataset cifar100 --crit ce --lr 0.025 --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_fairnas_fairsampling-cifar100_exp3.1_c100/graduate_nb201_fairnas_fairsampling-cifar100_exp3.1_c100_nb201_ckpt_0191.pth.tar'

# 第四章 实验3.2_c100 sandwich sampling cifar100
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_sandwich-sampling-cifar100_exp3.2_c100 --dataset cifar100 --crit ce --lr 0.025 --type sandwich
# 进化搜索
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_nb201_spos_sandwich-sampling-cifar100_exp3.2_c100 --dataset cifar100 --crit ce --lr 0.025 --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_spos_sandwich-sampling-cifar100_exp3.2_c100/graduate_nb201_spos_sandwich-sampling-cifar100_exp3.2_c100_nb201_ckpt_0191.pth.tar'


# 第四章 实验3.3_c100 balanced sampling with nwot cifar100
# TODO running
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_balanced-sampling-nwot-cifar100_exp3.3_c100_nwot --dataset cifar100 --crit ce --lr 0.025 --type nwot
# 进化搜索
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_nb201_spos_sandwich-sampling-cifar100_exp3.2_c100 --dataset cifar100 --crit ce --lr 0.025 --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_spos_balanced-sampling-nwot-cifar100_exp3.3_c100_nwot/graduate_nb201_spos_balanced-sampling-nwot-cifar100_exp3.3_c100_nwot_nb201_ckpt_0181.pth.tar'


# 第四章 实验3.0_img16 uniform sampling imagenet16
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_uniform-sampling-imagenet16_exp3.0_img16 --dataset imagenet16 --crit ce --lr 0.025 --type uniform --data_dir ./data/ImageNet16
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_nb201_spos_uniform-sampling-imagenet16_exp3.0_img16 --dataset imagenet16 --data_dir ./data/ImageNet16 --crit ce --lr 0.025 --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_spos_uniform-sampling-imagenet16_exp3.0_img16/graduate_nb201_spos_uniform-sampling-imagenet16_exp3.0_img16_nb201_ckpt_0191.pth.tar'

# 第四章 实验3.1_img16 fair sampling imagenet16
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_fairnas_fairsampling-imagenet16_exp3.1_img16 --dataset imagenet16 --crit ce --lr 0.025 --type fair --data_dir ./data/ImageNet16
# 进化搜索
# CUDA_VISIBLE_DEVICES=1 python tools/searcher/search_evo_nb201.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_nb201_fairnas_fairsampling-imagenet16_exp3.1_img16 --dataset imagenet16 --data_dir ./data/ImageNet16 --crit ce --lr 0.025 --model_path ''

# 第四章 实验3.2_img16 sandwich sampling imagenet16
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_sandwich-sampling-imagenet16-lr0.05_exp3.2_img16 --dataset imagenet16 --crit ce --lr 0.05 --type sandwich --data_dir ./data/ImageNet16
# 进化搜索
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_nb201_spos_sandwich-sampling-imagenet16_exp3.2_img16 --dataset imagenet16 --crit ce --data_dir ./data/ImageNet16 --lr 0.05 --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_spos_sandwich-sampling-imagenet16-lr0.05_exp3.2_img16/graduate_nb201_spos_sandwich-sampling-imagenet16-lr0.05_exp3.2_img16_nb201_ckpt_0161.pth.tar'

# 第四章 实验3.3_img16 balanced sampling with nwot imagenet16
# TODO running
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_spos_balanced-sampling-nwot-imagenet16_exp3.3_img16_nwot --dataset imagenet16 --crit ce --lr 0.025 --type nwot --data_dir ./data/ImageNet16
# 进化搜索
# CUDA_VISIBLE_DEVICES=0 python tools/searcher/search_evo_nb201.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name evo_nb201_spos_sandwich-sampling-imagenet16_exp3.3_img16 --dataset imagenet16 --crit ce --lr 0.025 --model_path '/home/stack/project/spos-cifar/checkpoints/graduate_nb201_spos_balanced-sampling-nwot-imagenet16_exp3.3_img16_nwot/graduate_nb201_spos_balanced-sampling-nwot-imagenet16_exp3.3_img16_nwot_nb201_ckpt_0191.pth.tar' --data_dir ./data/ImageNet16

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# 第五章 实验2.0 macro benchmark + random type
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_randomtype_exp2.0 --dataset cifar10 --crit ce --lr 0.025

# 第五章 实验2.1 macro benchmark + hamming type
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_hammingtype_exp2.1 --dataset cifar10 --crit ce --lr 0.025
# 调参 设置dis更小
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_hamming-type-smallerdis_exp2.1 --dataset cifar10 --crit ce --lr 0.025 --type hamming
# 调参 设置dis更大
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_hamming-type-largerdis_exp2.1 --dataset cifar10 --crit ce --lr 0.025 --type hamming


# 第五章 实验2.2 macro benchmark + adaptive hamming type
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_adaptivetype_exp2.2 --dataset cifar10 --crit ce --lr 0.025
# 调参 设置dis更小
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_adaptivetype-smallerdis_exp2.2 --dataset cifar10 --crit ce --lr 0.025
# 调参 设置dis更大
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_adaptivetype-largerdis_exp2.2 --dataset cifar10 --crit ce --lr 0.025


# 第五章 实验2.3 nb201 + adaptive hamming type
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_hammingtype_exp2.3 --dataset cifar10 --crit ce --lr 0.025
# TODO Reproduce
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_hammingtype_exp2.3_reproduce --dataset cifar10 --crit ce --lr 0.025 --type adaptive

# 第五章 实验2.3_nwot nb201 + adaptive hamming type
# TODOrunning 调参调到最高0.90，现在是
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_hamming-type_exp2.3_nwot --dataset cifar10 --crit ce --lr 0.025 --type adaptive
# TODO 调参 nwot lr=0.05
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_adaptive-hamming-type-lr0.05_exp2.3_nwot --dataset cifar10 --crit ce --lr 0.05 --type adaptive
# TODO 调参 nwot lr=0.1
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_adaptive-hamming-type-lr0.1_exp2.3_nwot --dataset cifar10 --crit ce --lr 0.1 --type adaptive
# TODO 调参 flops lr=0.05
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_pairwise_adaptive-hamming-type-lr0.05_exp2.3_flops --dataset cifar10 --crit ce --lr 0.05 --type pairwise
# TODO 调参 flops lr=0.1
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201_Balance_Trainer --log_name graduate_nb201_pairwise_adaptive-hamming-type-lr0.1_exp2.3_flops --dataset cifar10 --crit ce --lr 0.1 --type pairwise


# 第五章 实验2.4 nb201 + origin hamming type
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_origin-hamming-type_exp2.4 --dataset cifar10 --crit ce --lr 0.025

# 第五章 实验2.5 nb201 + random sample + CIFAR-100
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_random-type-cifar100_exp2.5 --dataset cifar100 --crit ce --lr 0.1

# 第五章 实验2.6 nb201 + random sample + ImageNet16
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_random-type-imagenet16_exp2.6 --dataset imagenet16 --crit ce --lr 0.1 --data_dir ./data/ImageNet16


# 第五章 实验2.7 Macro + random sample + CIFAR-100
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_adaptive-type-cifar100_exp2.7 --dataset cifar100 --crit ce --lr 0.025 --type random

# 第五章 实验2.7_hamming Macro + hamming sample + CIFAR-100
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_hamming-type-cifar100_exp2.7_hamming --dataset cifar100 --crit ce --lr 0.025 --type hamming

# 第五章 实验2.7_adaptive Macro + adaptive hamming sample + CIFAR-100
# CUDA_VISIBLE_DEVICES=2 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MacroBenchmarkSuperNet --trainer_name MacroTrainer --log_name graduate_macro_pairwise_adaptive-type-cifar100_exp2.7_adaptive --dataset cifar100 --crit ce --lr 0.025 --type adaptive


# 第五章 实验2.8 nb201 + hamming sample + CIFAR-100
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_hamming-type-cifar100_exp2.8 --dataset cifar100 --crit ce --lr 0.1 --type hamming


# 第五章 实验2.9 nb201 + hamming sample + ImageNet16
# CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_hamming-type-imagenet16_exp2.9 --dataset imagenet16 --crit ce --lr 0.1 --data_dir ./data/ImageNet16 --type hamming


# 第五章 实验2.10 nb201 + adaptive sample + CIFAR-100
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_adaptive-type-cifar100_exp2.10 --dataset cifar100 --crit ce --lr 0.1 --type adaptive


# 第五章 实验2.11 nb201 + adaptive sample + ImageNet16
# CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name graduate_nb201_pairwise_adaptive-type-imagenet16_exp2.11 --dataset imagenet16 --crit ce --lr 0.1 --data_dir ./data/ImageNet16 --type adaptive

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
