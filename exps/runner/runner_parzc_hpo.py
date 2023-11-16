import logging
import os
from argparse import ArgumentParser

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau

from piconas.core.losses.diffkd import diffkendall
from piconas.core.losses.landmark_loss import PairwiseRankLoss
from piconas.datasets.predictor.data_factory import create_dataloader
from piconas.predictor.pinat.model_factory import create_model_hpo
from piconas.utils.utils import AverageMeterGroup, accuracy_mse, set_seed, to_cuda
from piconas.utils.rank_consistency import spearman, pearson


parser = ArgumentParser()
# exp and dataset
parser.add_argument('--exp_name', type=str, default='ParZCBMM2')
parser.add_argument('--bench', type=str, default='101')
parser.add_argument('--train_split', type=str, default='172')
parser.add_argument('--eval_split', type=str, default='all')
parser.add_argument('--dataset', type=str, default='cifar10')
# training settings
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu_id', type=int, default=2)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--train_batch_size', default=20, type=int)
parser.add_argument('--eval_batch_size', default=512, type=int)
parser.add_argument('--train_print_freq', default=1e5, type=int)
parser.add_argument('--eval_print_freq', default=10, type=int)
parser.add_argument('--model_name', type=str, default='ParZCBMM2q')
args = parser.parse_args()

# Create a log directory if it doesn't exist
log_dir = './logdir'
os.makedirs(log_dir, exist_ok=True)

# initialize log info
log_format = '%(asctime)s %(message)s'
logging.basicConfig(
    filename=os.path.join(log_dir, 'training_hpo_parzc_main_4nb101.log'),
    level=logging.INFO,
    format=log_format,
    datefmt='%m/%d %I:%M:%S %p',
)
logging.info(args)

# set cpu/gpu device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# create dataloader and model
train_loader, test_loader, train_set, test_set = create_dataloader(args)


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
    assert args.train_split in train_splits
    assert args.eval_split in test_splits


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


def train(
    train_set,
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    criterion1: nn.MSELoss,
    criterion2: PairwiseRankLoss,
    epoch,
):
    model.train()
    for epoch in range(epoch):
        lr = optimizer.param_groups[0]['lr']
        meters = AverageMeterGroup()

        for step, batch in enumerate(train_loader):
            batch = to_cuda(batch, device)
            target = batch['val_acc']
            predict = model(batch)

            # Compute the losses
            # loss_mse = criterion1(predict, target.float())

            # loss_weight_1 = 0.95  # adjust as necessary
            # loss_weight_2 = 0.05  # adjust as necessary

            if len(predict.shape) > 1:  # for test only
                predict = predict.squeeze(-1)
            term3 = diffkendall(predict, target)
            loss = term3  # + 0.01 * term2


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For logging, we can compute MSE or other metrics if desired.
            meters.update(
                {'loss': loss.item(), }, n=target.size(0))

            if step % args.train_print_freq == 0 and epoch % 10 == 0:
                logging.info(
                    'Epoch [%d/%d] Step [%d/%d] lr = %.3e  %s',
                    epoch + 1,
                    args.epochs,
                    step + 1,
                    len(train_loader),
                    lr,
                    meters,
                )

        lr_scheduler.step()
    return model


def evaluate(test_set, test_loader, model, criterion):
    model.eval()
    meters = AverageMeterGroup()
    predict_list, target_list = [], []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            batch = to_cuda(batch, device)
            target = batch['test_acc']
            predict = model(batch)
            predict_list.append(predict.cpu().numpy())
            target_list.append(target.cpu().numpy())
            meters.update(
                {
                    'loss': criterion(predict, target).item(),
                    'mse': accuracy_mse(
                        predict.squeeze(), target.squeeze(), test_set
                    ).item(),
                },
                n=target.size(0),
            )
            if step % args.eval_print_freq == 0 or step + 1 == len(test_loader):
                logging.info(
                    'Evaluation Step [%d/%d]  %s', step +
                    1, len(test_loader), meters
                )
            # make np.array to str
            adj = batch['adjacency'].cpu().numpy()
            adj_str = np.array2string(adj)

    predict_list = np.concatenate(predict_list)
    target_list = np.concatenate(target_list)
    kendall_tau = kendalltau(predict_list, target_list)
    spearman_rho = spearman(predict_list, target_list)
    pearson_rho = pearson(predict_list, target_list)

    if isinstance(kendall_tau, tuple):
        kendall_tau = kendall_tau[0]
    if isinstance(spearman_rho, tuple):
        spearman_rho = spearman_rho[0]
    if isinstance(pearson_rho, tuple):
        pearson_rho = pearson_rho[0]


    return kendall_tau, spearman_rho, pearson_rho


def objective_function(hyperparameters):
    n_layers = hyperparameters['n_layers']
    n_head = hyperparameters['n_head']
    pine_hidden = hyperparameters['pine_hidden']
    linear_hidden = hyperparameters['d_word_model']
    d_word_model = hyperparameters['d_word_model']
    d_k_v = hyperparameters['d_k_v']
    d_inner = hyperparameters['d_inner']
    epoch = hyperparameters['epoch']
    dropout = hyperparameters['dropout']

    model = create_model_hpo(
        n_layers, n_head, pine_hidden, linear_hidden, d_word_model, d_k_v, d_inner
    )

    model = model.to(device)

    # define loss, optimizer, and lr_scheduler
    criterion1 = nn.MSELoss()
    criterion2 = PairwiseRankLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # train and evaluate predictor
    model = train(
        train_set,
        train_loader,
        model,
        optimizer,
        lr_scheduler,
        criterion1,
        criterion2,
        epoch=epoch,
    )
    kendall_tau, spearman_rho, pearson_rho = evaluate(
        test_set, test_loader, model, criterion1
    )

        # write results
    with open('./results/preds_%s.txt' % args.bench, 'a') as f:
        f.write(
            'EXP:%s\tlr: %s\ttrain: %s\ttest: %s\tkendall_tau: %.6f\t spearman_rho: %.6f\t pearson_rho: %.6f\n'
            % (
                args.exp_name,
                args.lr,
                args.train_split,
                args.eval_split,
                kendall_tau,
                spearman_rho,
                pearson_rho,
            )
        )

    return kendall_tau


def objective(trial):
    # Define the hyperparameters
    hyperparameters = {
        'n_layers': trial.suggest_int('n_layers', 2, 5),
        'n_head': trial.suggest_int('n_head', 3, 8),
        'pine_hidden': trial.suggest_int('pine_hidden', 8, 128),
        'd_word_model': trial.suggest_int('d_word_model', 256, 1024),
        'd_k_v': trial.suggest_int('d_k_v', 32, 128),
        'd_inner': trial.suggest_int('d_inner', 256, 1024),
        'epoch': trial.suggest_int('epoch', 20, 300),
        'dropout': trial.suggest_float('dropout', 0.01, 0.5),
    }

    # Use the objective function as you've defined it
    return objective_function(hyperparameters)


def main():
    check_arguments()

    # Setup the Optuna study, which will conduct the Bayesian optimization
    study = optuna.create_study(
        direction='maximize'
    )  # or 'minimize' if you're minimizing a loss
    study.optimize(objective, n_trials=100)

    # After the study is completed, you can get the best parameters and the best value (e.g., kendall tau)
    logging.info('Best HP: %s', study.best_params)
    logging.info('Best Kendalltau: %.6f', study.best_value)


if __name__ == '__main__':
    # run_func(args, main)
    main()
