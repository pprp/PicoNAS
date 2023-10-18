#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

echo "Workingdir: $PWD";
echo "Started at $(date)";


searchspace=${1:-nasbench201}
dataset=${2:-cifar10}

# 'jacov', 'snip', 'synflow', 'grad_norm', 'fisher', 'grasp', 'zen'
predictor=${3:-zen}
start_seed=${4:-9000}
experiment=${5:-only_zc}
seed=${6:-0}
optimizer=${7:-bananas}
N_MODELS=5

start=`date +%s`

seed=$(($start_seed + $seed))

# CUDA_VISIBLE_DEVICES=2 python -u exps/naslib/runner_cal_rc.py --config-file configs/${experiment}/${predictor}/${searchspace}-${start_seed}/${dataset}/config_${seed}.yaml

# CUDA_VISIBLE_DEVICES=2 python exps/naslib/runner_zc_ensemble.py --config-file configs/${experiment}/${optimizer}/${searchspace}-${start_seed}/${dataset}/config_${seed}.yaml

# benchmarks
# CUDA_VISIBLE_DEVICES=1 python exps/runner/runner_benchmark.py --config-file configs/${experiment}/${optimizer}/${searchspace}-${start_seed}/${dataset}/config_${start_seed}.yaml start_idx 0 n_models $N_MODELS predictor $predictor

CUDA_VISIBLE_DEVICES=1 python exps/ranker/zerocost_ranker.py  --model_name OneShotNASBench301Network --trainer_name NB301Trainer --log_name nb301_spos_test_x1 --dataset cifar10 --crit ce --lr 0.025

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
