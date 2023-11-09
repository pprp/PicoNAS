#!/bin/bash 

LOGDIR=./logdir/nds/rank 

if [ ! -d "$LOGDIR" ]; then
  mkdir -p $LOGDIR
fi

#  ['Amoeba', 'DARTS', 'ENAS', 'NASNet', 'PNAS']
SS_TYPE=(Amoeba DARTS ENAS NASNet PNAS)

IDX=1

for ((t=0; t<${#SS_TYPE[*]}; t++)); do
  # set gpu and data splits
  GPU=$((${IDX} % 8))
  let IDX+=1
  EXP_Name=${SS_TYPE[t]}_nds_rank_run2

  # run
  nohup python -u ./exps/eval_rank/eval_nds_rank.py --search_space ${SS_TYPE[t]} \
    --gpu_id $GPU \
    > $LOGDIR/$EXP_Name.log 2>&1 &

  echo "GPU:$GPU EXP:$EXP_Name"
  if [ $GPU = 7 ] ; then
      echo "sleep 30s"
      sleep 30s
  fi

done

tail -f $LOGDIR/$EXP_Name.log