#!/bin/bash

LOGDIR=logdir/rnd_search
mkdir -p $LOGDIR

ZC_NAME=eznas-a
IS_PREDICTOR=False
TAG=rndinput-run5
LOGFILE=$LOGDIR/$(date +%Y-%m-%d)-${ZC_NAME}-$TAG.log
CUDA_VISIBLE_DEVICES=3 python exps/searcher/search_rnd_nb201.py \
     --log_name rnd_search_$TAG \
     --is_predictor $IS_PREDICTOR \
     --measure_name $ZC_NAME > $LOGFILE 2>&1 &

tail -f $LOGFILE
