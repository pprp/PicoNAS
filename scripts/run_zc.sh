#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";


searchspace=${1:-nasbench201}
dataset=${2:-cifar10}
predictor=${3:-zen}
start_seed=${4:-0}
experiment=${5:-correlation}

start=`date +%s`

CUDA_VISIBLE_DEVICES=2 python -u tools/runner.py --config-file configs/${experiment}/${searchspace}-${dataset}-${predictor}-${start_seed}/config_${seed}.yaml 


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
