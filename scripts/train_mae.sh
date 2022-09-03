#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";
start=`date +%s`

for lr in 0.00625 0.0125 0.025 0.05 0.1
do
    echo "Current LR is $lr"
    # CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name SearchableMAE --trainer_name MAETrainer --log_name spos_mae_$lambda --dataset simmim --crit mse --lr $lr --epochs 100
    CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name MAESupernetNATS --trainer_name NATSMAETrainer --log_name nats_mae_lr_$lr --dataset simmim --crit mse --lr $lr --epochs 100
done


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
