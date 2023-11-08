#!/bin/bash

# logdir
logdir=logdir/nds
if [ ! -d $logdir ]; then
    mkdir -p $logdir
fi

# run ss=Amoeba cuda0
Log_name=nds_hpo_run1_$(date +%Y%m%d-%H%M%S)_cuda0_Amoeba_NDS.log
CUDA_VISIBLE_DEVICES=0 python exps/nds/runner_hpo.py --search_space Amoeba > $logdir/$Log_name 2>&1 &

# run ss=DARTS cuda1
Log_name=nds_hpo_run1_$(date +%Y%m%d-%H%M%S)_cuda1_DARTS_NDS.log
CUDA_VISIBLE_DEVICES=1 python exps/nds/runner_hpo.py --search_space DARTS > $logdir/$Log_name 2>&1 &

# run ss=ENAS cuda2
Log_name=nds_hpo_run1_$(date +%Y%m%d-%H%M%S)_cuda2_ENAS_NDS.log
CUDA_VISIBLE_DEVICES=2 python exps/nds/runner_hpo.py --search_space ENAS > $logdir/$Log_name 2>&1 &

# run ss=NASNet cuda3
Log_name=nds_hpo_run1_$(date +%Y%m%d-%H%M%S)_cuda3_NASNET_NDS.log
CUDA_VISIBLE_DEVICES=3 python exps/nds/runner_hpo.py --search_space NASNet > $logdir/$Log_name 2>&1 &

# run ss=PNAS cuda4
Log_name=nds_hpo_run1_$(date +%Y%m%d-%H%M%S)_cuda4_PNAS_NDS.log
CUDA_VISIBLE_DEVICES=4 python exps/nds/runner_hpo.py --search_space PNAS > $logdir/$Log_name 2>&1 &
