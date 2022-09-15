#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";
start=`date +%s`


CUDA_VISIBLE_DEVICES=1 python tools/train.py --config configs/spos/spos_cifar10.py --model_name OneShotNASBench201Network --trainer_name NB201Trainer --log_name test_nb201_spos --dataset cifar10 --crit ce --lr 0.025



end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime