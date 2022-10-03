#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python ./train_imagenet.py --config ../../configs/spos/spos_cifar10.py --model_name MobileNetv2MetaReceptionField --trainer_name MetaTrainer  --dataset cifar10 --crit ce --lr 0.05  --log_name meta_train_mbv2_rf_rerun --batch_size 64 --epochs 100 --sched cosine  --data_dir ../../data/cifar



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=8 ./train_timm.py /home/inspur/data/imagenet/  --model MobileNetv2MetaReceptionField --trainer MetaTrainer -b 128 --sched cosine --epochs 100 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.  --model-ema-decay 0.9999 --lr 0.1

# ./distributed_train.sh 2 /imagenet/ --model mobilenetv3_large_100 -b 512 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064 --lr-noise 0.42 0.9

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./train_timm.py /home/inspur/data/imagenet/  --model MobileNetv2MetaReceptionField --trainer MetaTrainer -b 128 --sched cosine --epochs 100 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.  --model-ema-decay 0.9999 --lr 0.1

# baseline: params: 3.5M  flops: 300M
# coordatt: params: 3.95M flops: 310M + 12% param
# se      : params: 3.89M flops: 300M +

# MobileNetv2MetaReceptionField v1:
# params: 17.52M flops: 1.27 GFlops

# MobileNetv2MetaReceptionField v1: 删掉strip pool
# params: 3.95M flops: 0.34 GFLOPS

# MobileNetv2MetaReceptionField v2: 删掉strip pool
# params: 3.63M flops: 0.33 GFLOPS

# 对齐实验设置。
# bs=64 x 8 GPUS
# opt=sgd with nesterov momentum
# lr=0.1
# sched=cosine
# epochs=150
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=8 ./train_timm.py /home/inspur/data/imagenet/  --model MobileNetv2MetaReceptionField --trainer MetaTrainer -b 64 --opt nesterov --momentum 0.9 --lr 0.1 --epochs 140 --sched cosine

# 如果不ok，可以用mobilenetv2进行蒸馏监督
# 遇到问题，所有架构都收敛到coord attention
#    解决思路： 1. print固定相应信息；2. 在subimagenet上进行实验; 3. 添加broadcast机制。
