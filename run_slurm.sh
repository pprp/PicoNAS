#!/bin/bash

module load anaconda
module load cuda/11.2
module load cudnn/8.1.0.77_CUDA11.2
source activate py38

# default is NB201 
python examples/gdnas/rank_gdnas.py --trainer_name GDNASTrainer --evaluator NB201Evaluator

# NB101 settings
python examples/gdnas/rank_gdnas.py --trainer_name GDNASTrainer --evaluator NB101Evaluator 
