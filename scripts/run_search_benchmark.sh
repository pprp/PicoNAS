#!/bin/bash

# This script runs the search benchmark for nasbench101 and nasbench201

LOGDIR=logdir/search_benchmark
if [ ! -d $LOGDIR ]; then
    mkdir -p $LOGDIR
fi

# NASBENCH101
# LOGFILE=$LOGDIR/search_benchmark_nasbench101.log
# python exps/searcher/search_benchmark.py --gpu_id 4 --train_split 100 --bench 101 > $LOGFILE 2>&1 &

# NASBENCH201
# DATASET=cifar10
# LOGFILE=$LOGDIR/search_benchmark_nasbench201_$DATASET.log
# python exps/searcher/search_benchmark.py --gpu_id 0 --train_split 78 --bench 201 --dataset $DATASET > $LOGFILE 2>&1 &

# DATASET=cifar100
# LOGFILE=$LOGDIR/search_benchmark_nasbench201_$DATASET.log
# python exps/searcher/search_benchmark.py --gpu_id 1 --train_split 78 --bench 201 --dataset $DATASET > $LOGFILE 2>&1 &


DATASET=ImageNet16-120
LOGFILE=$LOGDIR/search_benchmark_nasbench201_$DATASET.log
python exps/searcher/search_benchmark.py --gpu_id 2 --train_split 78 --bench 201 --dataset $DATASET > $LOGFILE 2>&1 &

tail -f $LOGFILE # show the log file
