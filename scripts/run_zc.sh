#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";


searchspace=${1:-nasbench201}
dataset=${2:-cifar10}
predictor=${3:-zen}
start_seed=${4:-9000}
experiment=${5:-only_zc}
seed=${6:-0}
optimizer=${7:-bananas}
N_MODELS=1

start=`date +%s`

seed=$(($start_seed + $seed))

# CUDA_VISIBLE_DEVICES=2 python -u tools/naslib/runner_cal_rc.py --config-file configs/${experiment}/${predictor}/${searchspace}-${start_seed}/${dataset}/config_${seed}.yaml

# CUDA_VISIBLE_DEVICES=2 python tools/naslib/runner_zc_ensemble.py --config-file configs/${experiment}/${optimizer}/${searchspace}-${start_seed}/${dataset}/config_${seed}.yaml

# benchmarks
CUDA_VISIBLE_DEVICES=1 python tools/naslib/runner_benchmark.py --config-file configs/${experiment}/${optimizer}/${searchspace}-${start_seed}/${dataset}/config_${start_seed}.yaml start_idx 0 n_models $N_MODELS

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime



end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
