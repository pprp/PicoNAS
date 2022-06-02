#!/bin/bash 


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

CUDA_VISIBLE_DEVICES=0 python -u tools/mae_pretrain.py