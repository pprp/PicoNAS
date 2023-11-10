#!/bin/bash

LOGDIR=./logdir/ablation/module_ablation
if [ ! -d "$LOGDIR" ]; then
  mkdir -p $LOGDIR
fi

# Loss ablation study
# LOSS_TYPE=(mse+pw+dk mse pairwise diffkendall mse+pw mse+dk pw+dk)

# IDX=1
# Train_Split=172
# Eval_Split=all
# Epochs=300
# Model=ParZCBMM
# Dataset=cifar10
# Train_batch_size=10
# Eval_batch_size=50
# Script=./piconas/runner/runner_pinat_rank.py
# Bench=101

# for((t=0; t<${#LOSS_TYPE[*]}; t++)); do
#   # set gpu and data splits
#   GPU=$((${IDX} % 8))
#   let IDX+=1
#   Loss=${LOSS_TYPE[t]}
#   EXP_Name=${Bench}_${Dataset}_${Model}_${Loss}_t${Train_Split}_v${Eval_Split}_e${Epochs}_bs${Train_batch_size}

#   # run
#   nohup python -u ${Script} --exp_name $EXP_Name --epochs $Epochs --gpu_id $GPU --model_name ${Model} \
#     --train_split ${Train_Split} --eval_split ${Eval_Split} --bench ${Bench} --dataset ${Dataset} \
#     --train_batch_size ${Train_batch_size} --eval_batch_size ${Eval_batch_size} \
#     > $LOGDIR/$EXP_Name.log 2>&1 &

#   echo "GPU:$GPU EXP:$EXP_Name"
#   if [ $GPU = 7 ] ; then
#       echo "sleep 30s"
#       sleep 30s
#   fi

# done

# tail -f $LOGDIR/$EXP_Name.log

# Module ablation study
MODULE_TYPE=(MLP NeuralPredictorMLPMixer NeuralPredictorBayesian BayesianNetwork MLPMixer NeuralPredictorModel)
IDX=1
Train_Split=100
Eval_Split=all
Epochs=300
Dataset=cifar10
Train_batch_size=10
Eval_batch_size=50
Script=./piconas/runner/runner_pinat_rank.py
Bench=101

for((t=0; t<${#MODULE_TYPE[*]}; t++)); do
  # set gpu and data splits
  GPU=$((${IDX} % 8))
  let IDX+=1
  Model=${MODULE_TYPE[t]}
  EXP_Name=${Bench}_${Dataset}_${Model}_t${Train_Split}_v${Eval_Split}_e${Epochs}_bs${Train_batch_size}_ABLATION_MODULE

  # run
  nohup python -u ${Script} --exp_name $EXP_Name --epochs $Epochs --gpu_id $GPU --model_name ${Model} \
    --train_split ${Train_Split} --eval_split ${Eval_Split} --bench ${Bench} --dataset ${Dataset} \
    --train_batch_size ${Train_batch_size} --eval_batch_size ${Eval_batch_size} \
    > $LOGDIR/$EXP_Name.log 2>&1 &

  echo "GPU:$GPU EXP:$EXP_Name"
  if [ $GPU = 7 ] ; then
      echo "sleep 30s"
      sleep 30s
  fi

done

tail -f $LOGDIR/$EXP_Name.log
