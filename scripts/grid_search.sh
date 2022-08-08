#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";
start=`date +%s`

for lambda in 0.0 0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5
do
    echo "Current Lambda is $lambda"
    CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name nb201_pair_$lambda --dataset cifar10 --crit ce --lr 0.025 --epochs 100 --p_lambda $lambda
done


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
