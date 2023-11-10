#!/bin/bash

LOGDIR=logdir/rnd_search
mkdir -p $LOGDIR

ZC_NAME=eznas-a
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-${ZC_NAME}-run1.log
CUDA_VISIBLE_DEVICES=0 python exps/searcher/search_rnd_nb201.py \
     --log_name rnd_search_run1 \
     --measure_name $ZC_NAME > $LOGFILE 2>&1 &

tail -f $LOGFILE
