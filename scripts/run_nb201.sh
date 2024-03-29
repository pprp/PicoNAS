#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";
start=`date +%s`

# CUDA_VISIBLE_DEVICES=1 python exps/train.py  --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name nb201_pairwiseloss --dataset cifar10 --crit ce --lr 0.025

CUDA_VISIBLE_DEVICES=1 python exps/train.py  --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name test_nb201_spos --dataset cifar10 --crit ce --lr 0.025

# CUDA_VISIBLE_DEVICES=1 python exps/train.py  --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name nb201_spos_w_sc --dataset cifar10 --crit ce --lr 0.025

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
