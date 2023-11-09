#!/bin/bash

LOGDIR=logdir/evo_search
mkdir -p $LOGDIR

LOGFILE=$LOGDIR/$(date +%Y-%m-%d_%H-%M-%S)-run1.log
CUDA_VISIBLE_DEVICES=2 python exps/searcher/search_evo_nb201.py --log_name evo_search_run1 > $LOGFILE 2>&1 &

LOGFILE=$LOGDIR/$(date +%Y-%m-%d_%H-%M-%S)-run2.log
CUDA_VISIBLE_DEVICES=3 python exps/searcher/search_evo_nb201.py --log_name evo_search_run2 > $LOGFILE 2>&1 &

LOGFILE=$LOGDIR/$(date +%Y-%m-%d_%H-%M-%S)-run3.log
CUDA_VISIBLE_DEVICES=4 python exps/searcher/search_evo_nb201.py --log_name evo_search_run3 > $LOGFILE 2>&1 &

tail -f $LOGFILE
