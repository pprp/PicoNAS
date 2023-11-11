#!/bin/bash

LOGDIR=work_dir/one_shot_predictor
if [ ! -d $LOGDIR ]; then
    mkdir -p $LOGDIR
fi

# CUDA_VISIBLE_DEVICES=0 python train_rdnas_nb201.py \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 4 \
#     --data_dir /data/lujunl/pprp/bench/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 6 \
#     --log_name spos_train_valid_c10_after_predictor_seed6_run0_50samples
    # > ./work_dir/spos_train_valid_c10_after_predictor_seed6_run0_50samples.log 2>&1

LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-spos_train_valid_c10_after_predictor_seed42_run0_20samples.log
CUDA_VISIBLE_DEVICES=2 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs 250 --batch_size 16 \
    --data_dir /data/lujunl/pprp/bench/cifar --dataset cifar10 \
    --type adaptive  \
    --seed 42 \
    --log_name spos_train_valid_c10_after_predictor_seed42_run0_20samples > $LOGFILE 2>&1 &

tail -f $LOGFILE
    
# ------------------------ train pgonas with train-valid
# CUDA_VISIBLE_DEVICES=1 python train_rdnas.py \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 6 \
#     --log_name spos_train_valid_c10_seed6_run0

# CUDA_VISIBLE_DEVICES=0 python train_rdnas.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 66 \
#     --log_name spos_train_valid_c10_seed66_run1

# CUDA_VISIBLE_DEVICES=0 python train_rdnas.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name PGONASTrainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 666 \
#     --log_name spos_train_valid_c10_seed666_run2

# ------------------------- train pgo-nas with train-test
# CUDA_VISIBLE_DEVICES=0 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 6 \
#     --log_name spos_train_test_c10_seed6_run0

# CUDA_VISIBLE_DEVICES=0 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 66 \
#     --log_name spos_train_test_c10_seed66_run1

# CUDA_VISIBLE_DEVICES=0 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 666 \
#     --log_name spos_train_test_c10_seed666_run2

# CUDA_VISIBLE_DEVICES=1 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 42 \
#     --log_name spos_train_test_c10_seed42_run3

# CUDA_VISIBLE_DEVICES=1 python ../../exps/train.py \
#      \
#     --model_name OneShotNASBench201Network \
#     --trainer_name NB201Trainer \
#     --crit ce --lr 0.025 \
#     --epochs 250 --batch_size 64 \
#     --data_dir ../../data/cifar --dataset cifar10 \
#     --type adaptive  \
#     --seed 43 \
#     --log_name spos_train_test_c10_seed43_run4
