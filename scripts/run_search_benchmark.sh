#!/bin/bash

# This script runs the search benchmark for nasbench101 and nasbench201

LOGDIR=logdir/search_benchmark
if [ ! -d $LOGDIR ]; then
    mkdir -p $LOGDIR
fi

# NASBENCH101
LOGFILE=$LOGDIR/search_benchmark_nasbench101.log
python exps/searcher/search_benchmark.py --gpu_id 4 --train_split 100 --bench 101 > $LOGFILE 2>&1 &

# NASBENCH201
LOGFILE=$LOGDIR/search_benchmark_nasbench201.log
python exps/searcher/search_benchmark.py --gpu_id 5 --train_split 78 --bench 201 > $LOGFILE 2>&1 &
