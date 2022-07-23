#!/bin/bash

CKPT_PATH="./checkpoints/log_spos_c10_train_supernet_retrain_epoch600_super_ckpt_0100.pth.tar"
TYPE='test_acc'
NUM_SAMPLE=50
LOGNAME="eval_100e_type_test_acc_50sample.log"


echo $LOGNAME > logdir/${LOGNAME}
python tools/eval_rank.py --ckpt-path $CKPT_PATH --type $TYPE --num-sample $NUM_SAMPLE > logdir/${LOGNAME} 2>&1 &

sleep 3s

tail -f logdir/${LOGNAME}

# sample 10: 24.4
# sample 20: 14.24
# sample 50: -0.0099
