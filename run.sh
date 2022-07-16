#!/usr/bin/env sh

NUM_GPUS=1

MKL_NUM_THREADS=4
OMP_NUM_THREADS=1

srun --partition=mm_model \
    --job-name=autoformer_reimplement \
    --gres=gpu:$NUM_GPUS \
    --ntasks=$NUM_GPUS \
    --ntasks-per-node=$NUM_GPUS \
    --cpus-per-task=$NUM_GPUS \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    --async \
    python
