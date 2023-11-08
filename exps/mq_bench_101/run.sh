#!/bin/bash


if [ ! -d "./logdir/mq_bench_101" ]; then
  mkdir ./logdir/mq_bench_101
fi


# CUDA_VISIBLE_DEVICES=7 python exps/mq_bench_101/runner_hpo.py > logdir/mq_bench_101/runner_hpo_run0.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python exps/mq_bench_101/runner_hpo.py > logdir/mq_bench_101/runner_hpo_run1.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python exps/mq_bench_101/runner_hpo.py > logdir/mq_bench_101/runner_hpo_run2.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python exps/mq_bench_101/runner_hpo.py > logdir/mq_bench_101/runner_hpo_run3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python exps/mq_bench_101/runner_hpo.py > logdir/mq_bench_101/runner_hpo_run4.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python exps/mq_bench_101/runner_hpo.py > logdir/mq_bench_101/runner_hpo_run5.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python exps/mq_bench_101/runner_hpo.py > logdir/mq_bench_101/runner_hpo_run6.log 2>&1 &
