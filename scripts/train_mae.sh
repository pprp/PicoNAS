#!/bin/bash


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

CUDA_VISIBLE_DEVICES=2 python -u tools/mae_pretrain.py 2>&1 &

sleep 3s

tail -f logdir/${LOGNAME}
