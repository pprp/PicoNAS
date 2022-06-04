# Auto running for training supernet using SPOS

if [ ! -d "./logdir" ]; then
  mkdir ./logdir
fi

LogName=log_spos_c10_train_supernet_retrain
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train_macro.py --exp_name ${LogName} > logdir/${LogName} 2>&1 &

sleep 3s

tail -f logdir/${LogName}
