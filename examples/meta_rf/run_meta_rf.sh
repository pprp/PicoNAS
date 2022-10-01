#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python ./train_imagenet.py --config ../../configs/spos/spos_cifar10.py --model_name MobileNetv2MetaReceptionField --trainer_name MetaTrainer  --dataset cifar10 --crit ce --lr 0.05  --log_name meta_train_mbv2_rf_rerun --batch_size 64 --epochs 100 --sched cosine  --data_dir ../../data/cifar



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=8 ./train_timm.py /home/inspur/data/imagenet/  --model MobileNetv2MetaReceptionField --trainer MetaTrainer -b 128 --sched cosine --epochs 100 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.  --model-ema-decay 0.9999 --lr 0.1

# ./distributed_train.sh 2 /imagenet/ --model mobilenetv3_large_100 -b 512 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064 --lr-noise 0.42 0.9

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./train_timm.py /home/inspur/data/imagenet/  --model MobileNetv2MetaReceptionField --trainer MetaTrainer -b 128 --sched cosine --epochs 100 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.  --model-ema-decay 0.9999 --lr 0.1
