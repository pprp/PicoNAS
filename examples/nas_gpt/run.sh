#!/bin/bash

module load anaconda
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
source activate py38

python train.py
