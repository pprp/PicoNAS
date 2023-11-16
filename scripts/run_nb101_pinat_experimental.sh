#!/bin/bash

LOGDIR=./logdir/nb101
if [ ! -d "./logdir/nb101" ]; then
  mkdir ./logdir/nb101
fi
if [ ! -d "./checkpoints" ]; then
  mkdir ./checkpoints
fi
if [ ! -d "./results" ]; then
  mkdir ./results
fi

# arguments
IDX=3
Loss=pairwise
Bench=101
Epochs=124
Model=ParZCBMM2
Dataset=cifar10
Train_batch_size=10
Eval_batch_size=50
# Train_Split_List=(100 172 424 424 4236)
# Eval_Split_List=(all all 100 all all)
Train_Split_List=(172)
Eval_Split_List=(all)
Script=./exps/runner/runner_pinat_rank.py
SEED=555

for((t=0; t<${#Train_Split_List[*]}; t++)); do
  # set gpu and data splits
  GPU=$((${IDX} % 8))
  let IDX+=1
  Train_Split=${Train_Split_List[t]}
  Eval_Split=${Eval_Split_List[t]}
  EXP_Name=${Bench}_${Dataset}_${Model}_${Loss}_t${Train_Split}_v${Eval_Split}_e${Epochs}_bs${Train_batch_size}_seed${SEED}

  # run
  # nohup python -u ${Script} --exp_name $EXP_Name --epochs $Epochs --gpu_id $GPU --model_name ${Model} \
  #   --train_split ${Train_Split} --eval_split ${Eval_Split} --bench ${Bench} --dataset ${Dataset} \
  #   --train_batch_size ${Train_batch_size} --eval_batch_size ${Eval_batch_size} --seed ${SEED} \
  #   --loss_type ${Loss} \
  #   > $LOGDIR/$EXP_Name.log 2>&1 &
  # echo EXP_Name: $EXP_Name
  # # debug
  python -u ${Script} --exp_name $EXP_Name --epochs $Epochs --gpu_id $GPU --model_name ${Model} \
    --train_split ${Train_Split} --eval_split ${Eval_Split} --bench ${Bench} --dataset ${Dataset} \
    --train_batch_size ${Train_batch_size} --eval_batch_size ${Eval_batch_size} --loss_type ${Loss}

  echo "GPU:$GPU EXP:$EXP_Name"
  if [ $GPU = 7 ] ; then
      echo "sleep 30s"
      sleep 30s
  fi

done

tail -f $LOGDIR/$EXP_Name.log
