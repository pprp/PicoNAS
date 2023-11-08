# Auto running for training supernet using SPOS

if [ ! -d "./logdir" ]; then
  mkdir ./logdir
fi

LogName=log_spos_c10_train_supernet_retrain_epoch600
CUDA_VISIBLE_DEVICES=0 nohup python -u exps/train_macro.py --exp_name ${LogName} --epochs 600 >> logdir/${LogName} 2>&1 &

sleep 3s

tail -f logdir/${LogName}
