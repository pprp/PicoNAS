# Auto running for training supernet using SPOS

if [ ! -d "./logdir" ]; then
  mkdir ./logdir
fi

LogName=log_spos_c10_train_supernet
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/search_supernet.py --exp_name logName > logdir/${LogName} 2>&1 &

sleep 3s

tail -f logdir/${LogName}
