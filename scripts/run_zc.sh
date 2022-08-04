#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";


searchspace=${1:-nasbench201}
dataset=${2:-cifar10}
predictor=${3:-zen}
start_seed=${4:-9000}
experiment=${5:-correlation}
seed=${6:-0}

start=`date +%s`

seed=$(($start_seed + $seed))

CUDA_VISIBLE_DEVICES=2 python -u tools/runner.py --config-file configs/${experiment}/${predictor}/${searchspace}-${start_seed}/${dataset}/config_${seed}.yaml


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
