#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";
start=`date +%s`

for lambda in 0.02 0.04 0.06 0.08 0.10
do
    echo "Current Lambda is $lambda"
    CUDA_VISIBLE_DEVICES=0 python exps/train.py  --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name nb201_pair_$lambda --dataset cifar10 --crit ce --lr 0.025 --epochs 100 --p_lambda $lambda
done


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
