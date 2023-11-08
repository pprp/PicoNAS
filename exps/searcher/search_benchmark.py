# find the best architecture based on the predictor

import logging
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau

from piconas.core.losses.landmark_loss import PairwiseRankLoss
from piconas.datasets.predictor.data_factory import create_dataloader
from piconas.predictor.pinat.model_factory import (create_best_nb101_model,
                                                   create_model,
                                                   create_nb201_model)
from piconas.utils.utils import (AverageMeterGroup, accuracy_mse, set_seed,
                                 to_cuda)

parser = ArgumentParser()
# exp and dataset
parser.add_argument(
    '--exp_name', type=str, default='Search Benchmark for best accuracy')
parser.add_argument('--bench', type=str, default='101')
parser.add_argument('--train_split', type=str, default='100')
parser.add_argument('--eval_split', type=str, default='all')
parser.add_argument('--dataset', type=str, default='cifar10')
# training settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--train_batch_size', default=10, type=int)
parser.add_argument('--eval_batch_size', default=512, type=int)
parser.add_argument('--train_print_freq', default=1e5, type=int)
parser.add_argument('--eval_print_freq', default=10, type=int)
parser.add_argument('--model_name', type=str, default='ParZCBMM')
args = parser.parse_args()

# initialize log info
log_format = '%(asctime)s %(message)s'
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt='%m/%d %I:%M:%S %p')
logging.info(args)

# set cpu/gpu device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def check_arguments():
    # set seed
    set_seed(args.seed)

    # check ckpt and results dir
    assert args.exp_name is not None
    ckpt_dir = './checkpoints/nasbench_%s/' % args.bench
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    res_dir = './results/'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    # check data split
    if args.bench == '101':
        train_splits = ['100', '172', '424', '4236']
        test_splits = ['100', 'all']
    elif args.bench == '201':
        train_splits = ['78', '156', '469', '781', '1563']
        test_splits = ['all']
    else:
        raise ValueError('No defined NAS bench!')
    assert args.train_split in train_splits, f'{args.train_split} not in {train_splits}'
    assert args.eval_split in test_splits, f'{args.eval_split} not in {test_splits}'


def pair_loss(outputs, labels):
    output = outputs.unsqueeze(1)
    output1 = output.repeat(1, outputs.shape[0])
    label = labels.unsqueeze(1)
    label1 = label.repeat(1, labels.shape[0])
    tmp = (output1 - output1.t()) * torch.sign(label1 - label1.t())
    tmp = torch.log(1 + torch.exp(-tmp))
    eye_tmp = tmp * torch.eye(len(tmp)).cuda()
    new_tmp = tmp - eye_tmp
    loss = torch.sum(new_tmp) / (outputs.shape[0] * (outputs.shape[0] - 1))
    return loss


def evaluate(test_set, test_loader, model, criterion):
    model.eval()
    meters = AverageMeterGroup()
    predicts, targets = [], []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            batch = to_cuda(batch, device)
            target = batch['test_acc_wo_normalize']
            predict = model(batch)
            predicts.append(predict.cpu().numpy())
            targets.append(target.cpu().numpy())
            meters.update(
                {
                    'loss':
                    criterion(predict, target).item(),
                    'mse':
                    accuracy_mse(predict.squeeze(), target.squeeze(),
                                 test_set).item()
                },
                n=target.size(0))
            if step % args.eval_print_freq == 0 or step + 1 == len(
                    test_loader):
                logging.info('Evaluation Step [%d/%d]  %s', step + 1,
                             len(test_loader), meters)

    predicts = np.concatenate(predicts)
    targets = np.concatenate(targets)
    kendall_tau = kendalltau(predicts, targets)[0]
    return kendall_tau, predicts, targets


def traverse_benchmark(test_set, test_loader, model, topN=10):
    model.eval()
    meters = AverageMeterGroup()
    predicts, val_target_list, test_target_list = [], [], []

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            batch = to_cuda(batch, device)
            test_target = batch['test_acc_wo_normalize']
            val_target = batch['val_acc_wo_normalize']
            predict = model(batch)
            predicts.append(predict.cpu().numpy())
            test_target_list.append(test_target.cpu().numpy())
            val_target_list.append(val_target.cpu().numpy())

    predicts = np.concatenate(predicts)
    val_target_list = np.concatenate(val_target_list)
    test_target_list = np.concatenate(test_target_list)

    # plot correlation between predicts and targets
    import matplotlib.pyplot as plt
    plt.scatter(predicts, val_target_list)
    plt.savefig('correlation-w-valacc-vs-parzc.png')
    plt.clf()

    plt.scatter(predicts, test_target_list)
    plt.savefig('correlation-w-testacc-vs-parzc.png')
    plt.clf()

    # find the topN high score's index from predicts and then find the corresponding targest
    top_indices = np.argsort(predicts)[-topN:]
    topN_val_targets = val_target_list[top_indices]
    best_val_target = max(topN_val_targets)

    plt.scatter(predicts[top_indices], val_target_list[top_indices])
    plt.savefig('correlation-w-valacc-vs-parzc-top.png')

    topN_test_targets = test_target_list[top_indices]
    best_test_target = max(topN_test_targets)
    plt.clf()

    plt.scatter(predicts[top_indices], test_target_list[top_indices])
    plt.savefig('correlation-w-testacc-vs-parzc-top.png')
    plt.clf()

    logging.info(
        'Currently we found that the best val acc of search space is:',
        best_val_target)
    logging.info(
        'Currently we found that the best test acc of search space is:',
        best_test_target)

    # find highest idx for predicts
    highest_idx = np.argmax(predicts)
    norm_best_val_acc = val_target_list[highest_idx][0]
    norm_best_test_acc = test_target_list[highest_idx][0]

    logging.info(
        'Currently we found that the best val acc of search space is:',
        norm_best_val_acc)
    logging.info(
        'Currently we found that the best test acc of search space is:',
        norm_best_test_acc)

    return best_val_target, best_test_target


def main():
    check_arguments()

    # create dataloader and model
    train_loader, test_loader, train_set, test_set = create_dataloader(args)

    if args.bench == '201':
        model = create_nb201_model()
        ckpt_dir = 'checkpoints/nasbench_201/201_cifar10_PINATModel7_mse_t1563_vall_e300_bs10_final_tau0.792052_ckpt.pt'
    elif args.bench == '101':
        model = create_best_nb101_model()
        ckpt_dir = 'checkpoints/nasbench_101/101_cifar10_ParZCBMM_mse_t100_vall_e300_bs10_best_nb101_run0_tau0.648251_ckpt.pt'

    model.load_state_dict(
        torch.load(ckpt_dir, map_location=torch.device('cpu')))

    model = model.to(device)
    # logging.info(model)
    logging.info('PINAT params.: %f M' %
                 (sum(_param.numel() for _param in model.parameters()) / 1e6))
    logging.info('Training on NAS-Bench-%s, train_split: %s, eval_split: %s' %
                 (args.bench, args.train_split, args.eval_split))

    # define loss, optimizer, and lr_scheduler
    criterion1 = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # train and evaluate predictorß
    kendall_tau, predict_all, target_all = evaluate(test_set, test_loader,
                                                    model, criterion1)
    logging.info('Kendalltau: %.6f', kendall_tau)

    # evaluate the final performance across the whole dataset
    best_val_acc, best_test_acc = traverse_benchmark(
        test_set, test_loader, model, topN=5)


if __name__ == '__main__':
    # run_func(args, main)
    main()
