#!/bin/bash

LOGDIR=work_dir/one_shot_predictor
if [ ! -d $LOGDIR ]; then
    mkdir -p $LOGDIR
fi

BS=64
SEED=42
EPOCH=250
TAG=run0-baseline
LOGNAME=spos_train_valid_c10_after_predictor_bs${BS}_seed${SEED}_epoch${EPOCH}_${TAG}_20samples
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-$LOGNAME.log
CUDA_VISIBLE_DEVICES=0 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs $EPOCH \
    --batch_size $BS \
    --data_dir /data/lujunl/pprp/bench/cifar \
    --dataset cifar10 \
    --type adaptive  \
    --seed $SEED \
    --log_name ${LOGNAME}.log > $LOGFILE 2>&1 &


BS=32
SEED=42
EPOCH=250
TAG=run1-bs32
LOGNAME=spos_train_valid_c10_after_predictor_bs${BS}_seed${SEED}_epoch${EPOCH}_${TAG}_20samples
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-$LOGNAME.log
CUDA_VISIBLE_DEVICES=1 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs $EPOCH \
    --batch_size $BS \
    --data_dir /data/lujunl/pprp/bench/cifar \
    --dataset cifar10 \
    --type adaptive  \
    --seed $SEED \
    --log_name ${LOGNAME}.log > $LOGFILE 2>&1 &


BS=64
SEED=42
EPOCH=100
TAG=run2-ep100
LOGNAME=spos_train_valid_c10_after_predictor_bs${BS}_seed${SEED}_epoch${EPOCH}_${TAG}_20samples
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-$LOGNAME.log
CUDA_VISIBLE_DEVICES=2 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs $EPOCH \
    --batch_size $BS \
    --data_dir /data/lujunl/pprp/bench/cifar \
    --dataset cifar10 \
    --type adaptive  \
    --seed $SEED \
    --log_name ${LOGNAME}.log > $LOGFILE 2>&1 &

BS=64
SEED=42
EPOCH=300
TAG=run3-ep300
LOGNAME=spos_train_valid_c10_after_predictor_bs${BS}_seed${SEED}_epoch${EPOCH}_${TAG}_20samples
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-$LOGNAME.log
CUDA_VISIBLE_DEVICES=3 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs $EPOCH \
    --batch_size $BS \
    --data_dir /data/lujunl/pprp/bench/cifar \
    --dataset cifar10 \
    --type adaptive  \
    --seed $SEED \
    --log_name ${LOGNAME}.log > $LOGFILE 2>&1 &

BS=64
SEED=666
EPOCH=250
TAG=run4-sd666
LOGNAME=spos_train_valid_c10_after_predictor_bs${BS}_seed${SEED}_epoch${EPOCH}_${TAG}_20samples
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-$LOGNAME.log
CUDA_VISIBLE_DEVICES=4 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs $EPOCH \
    --batch_size $BS \
    --data_dir /data/lujunl/pprp/bench/cifar \
    --dataset cifar10 \
    --type adaptive  \
    --seed $SEED \
    --log_name ${LOGNAME}.log > $LOGFILE 2>&1 &

BS=16
SEED=42
EPOCH=250
TAG=run5-bs16
LOGNAME=spos_train_valid_c10_after_predictor_bs${BS}_seed${SEED}_epoch${EPOCH}_${TAG}_20samples
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-$LOGNAME.log
CUDA_VISIBLE_DEVICES=5 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs $EPOCH \
    --batch_size $BS \
    --data_dir /data/lujunl/pprp/bench/cifar \
    --dataset cifar10 \
    --type adaptive  \
    --seed $SEED \
    --log_name ${LOGNAME}.log > $LOGFILE 2>&1 &



BS=64
SEED=42
EPOCH=50
TAG=run6-ep50
LOGNAME=spos_train_valid_c10_after_predictor_bs${BS}_seed${SEED}_epoch${EPOCH}_${TAG}_20samples
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-$LOGNAME.log
CUDA_VISIBLE_DEVICES=6 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs $EPOCH \
    --batch_size $BS \
    --data_dir /data/lujunl/pprp/bench/cifar \
    --dataset cifar10 \
    --type adaptive  \
    --seed $SEED \
    --log_name ${LOGNAME}.log > $LOGFILE 2>&1 &



BS=64
SEED=3407
EPOCH=250
TAG=run7-sd3407
LOGNAME=spos_train_valid_c10_after_predictor_bs${BS}_seed${SEED}_epoch${EPOCH}_${TAG}_20samples
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-$LOGNAME.log
CUDA_VISIBLE_DEVICES=7 python train_rdnas.py \
    --model_name OneShotNASBench201Network \
    --trainer_name PGONASTrainer \
    --crit ce --lr 0.025 \
    --epochs $EPOCH \
    --batch_size $BS \
    --data_dir /data/lujunl/pprp/bench/cifar \
    --dataset cifar10 \
    --type adaptive  \
    --seed $SEED \
    --log_name ${LOGNAME}.log > $LOGFILE 2>&1 &

tail -f $LOGFILE
