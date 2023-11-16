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
from piconas.predictor.pinat.model_factory import (create_best_nb201_model,
                                                   create_model,
                                                   create_nb201_model)
from piconas.utils.utils import (AverageMeterGroup, accuracy_mse, set_seed,
                                 to_cuda)

parser = ArgumentParser()
# exp and dataset
parser.add_argument('--exp_name', type=str, default='PINAT')
parser.add_argument('--bench', type=str, default='201')
parser.add_argument('--train_split', type=str, default='78')
parser.add_argument('--eval_split', type=str, default='all')
parser.add_argument('--dataset', type=str, default='cifar10')
# training settings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--train_batch_size', default=10, type=int)
parser.add_argument('--eval_batch_size', default=10240, type=int)
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
            target = batch['test_acc']
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

    # save predicts and targets to 'correlation_parzc.csv'
    file_path = 'correlation_parzc.csv'
    with open(file_path, 'w') as f:
        f.write('predicts,targets\n')
        for i in range(len(predicts)):
            f.write('{},{}\n'.format(predicts[i], targets[i]))

    import matplotlib.pyplot as plt

    # filter out architectures gt < -5
    predicts = predicts[targets > -5]
    targets = targets[targets > -5]

    plt.scatter(
        predicts,
        targets,
        alpha=0.3,
        s=5,
        label='kendall_tau: %.4f' % kendall_tau)

    # Label and title
    plt.xlabel('Predicted Performance')
    plt.ylabel('Ground Truth Performance')
    plt.title('Correlation between Predicted and Ground Truth Performance')

    # Adjust axis limits
    plt.xlim(min(predicts), max(predicts))
    plt.ylim(min(targets), max(targets))

    # Add a legend
    plt.legend()

    # Save the figure
    plt.savefig('scatterplot.png')
    plt.close()

    # filter the top 10% architectures
    top_idx = np.argsort(predicts)[-int(len(predicts) * 0.05):]
    predicts = predicts[top_idx]
    targets = targets[top_idx]

    plt.scatter(
        predicts,
        targets,
        alpha=0.3,
        s=5,
        label='kendall_tau: %.4f' % kendall_tau)

    # Label and title
    plt.xlabel('Predicted Performance')
    plt.ylabel('Ground Truth Performance')
    plt.title('Correlation between Predicted and Ground Truth Performance')

    # Adjust axis limits
    plt.xlim(min(predicts), max(predicts))
    plt.ylim(min(targets), max(targets))

    # Add a legend
    plt.legend()

    # Save the figure
    plt.savefig('scatterplot_top.png')

    return kendall_tau, predicts, targets


def main():
    check_arguments()

    # create dataloader and model
    train_loader, test_loader, train_set, test_set = create_dataloader(args)
    # model = create_model(args)
    # model = create_nb201_model()

    model = create_best_nb201_model()

    # load model
    ckpt_dir = 'checkpoints/nasbench_201/201_cifar10_ParZCBMM_mse_t781_vall_e153_bs10_best_nb201_run2_tau0.783145_ckpt.pt'
    model.load_state_dict(
        torch.load(ckpt_dir, map_location=torch.device('cpu')))

    model = model.to(device)
    print(model)
    logging.info('PINAT params.: %f M' %
                 (sum(_param.numel() for _param in model.parameters()) / 1e6))
    logging.info('Training on NAS-Bench-%s, train_split: %s, eval_split: %s' %
                 (args.bench, args.train_split, args.eval_split))

    # define loss, optimizer, and lr_scheduler
    criterion1 = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # train and evaluate predictor
    kendall_tau, predict_all, target_all = evaluate(test_set, test_loader,
                                                    model, criterion1)
    logging.info('Kendalltau: %.6f', kendall_tau)

    # save checkpoint
    ckpt_dir = './checkpoints/nasbench_%s/' % args.bench
    ckpt_path = os.path.join(
        ckpt_dir, '%s_tau%.6f_ckpt.pt' % (args.exp_name, kendall_tau))
    torch.save(model.state_dict(), ckpt_path)
    logging.info('Save model to %s' % ckpt_path)

    # write results
    with open('./results/preds_%s.txt' % args.bench, 'a') as f:
        f.write('EXP:%s\tlr: %s\ttrain: %s\ttest: %s\tkendall_tau: %.6f\n' %
                (args.exp_name, args.lr, args.train_split, args.eval_split,
                 kendall_tau))


if __name__ == '__main__':
    # run_func(args, main)
    main()
