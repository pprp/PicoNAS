if [ ! -d "./logdir/nb201" ]; then
  mkdir ./logdir/nb201
fi
if [ ! -d "./checkpoints" ]; then
  mkdir ./checkpoints
fi
if [ ! -d "./results" ]; then
  mkdir ./results
fi

Dataset_List=(ImageNet16-120)
# arguments
IDX=0
Loss=mse
Bench=201
Epochs=100
Model=ParZCBMM
# Dataset=cifar10
Train_batch_size=10
Eval_batch_size=512
# Train_Split_List=(1563 781 469 156 78)
# Eval_Split_List=(all all all all all)
Train_Split_List=(78)
Eval_Split_List=(all)
# Script=./piconas/runner/runner_pinat.py
Script=./piconas/runner/runner_pinat_rank.py

python -u ${Script} --exp_name testing --epochs $Epochs --gpu_id $IDX --model_name ${Model} \
    --train_split 78 --eval_split all --bench ${Bench} --dataset ${Dataset} \
    --train_batch_size 32 --eval_batch_size 512


# for((t=0; t<${#Train_Split_List[*]}; t++)) do
#   # tranverse all the dataset_list
#   for ((j=0; j<${#Dataset_List[*]}; j++)) do
#     # set gpu and data splits
#     GPU=$((${IDX} % 8))
#     let IDX+=1
#     Train_Split=${Train_Split_List[t]}
#     Eval_Split=${Eval_Split_List[t]}
#     Dataset=${Dataset_List[j]}
#     EXP_Name=${Bench}_${Dataset}_${Model}_${Loss}_t${Train_Split}_v${Eval_Split}_e${Epochs}_bs${Train_batch_size}_best_nb201_RUN_NB201KD

#     # run
#     nohup python -u ${Script} --exp_name $EXP_Name --epochs $Epochs --gpu_id $GPU --model_name ${Model} \
#       --train_split ${Train_Split} --eval_split ${Eval_Split} --bench ${Bench} --dataset ${Dataset} \
#       --train_batch_size ${Train_batch_size} --eval_batch_size ${Eval_batch_size} \
#       >> logdir/nb201/$EXP_Name.log 2>&1 &

#   echo "GPU:$GPU EXP:$EXP_Name"
#   if [ $GPU = 7 ] ; then
#       echo "sleep 30s"
#       sleep 30s
#   fi
#   done
# done
# tail -f logdir/$EXP_Name.log
