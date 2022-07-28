import argparse
import os
import time

import init_paths  # noqa: F401
import torch
import torch.nn as nn

import pplib.utils.utils as utils
from pplib.datasets import build_loader_simmim
from pplib.models import build_model
from pplib.nas.mutators import OneShotMutator
from pplib.trainer import build_trainer
from pplib.utils.config import Config
from pplib.utils.logging import get_logger


def get_args():
    parser = argparse.ArgumentParser('train macro benchmark')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/spos/spos_cifar10.py',
        required=True,
        help='user settings config')
    parser.add_argument(
        '--exp_name',
        type=str,
        default='macro_cifar10',
        help='experiment name')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/cifar',
        help='path to the dataset')

    parser.add_argument(
        '--model_name',
        type=str,
        default='MAESupernetNATS',
        help='name of model')
    parser.add_argument(
        '--trainer_name',
        type=str,
        default='NATSMAETrainer',
        help='name of trainer')
    parser.add_argument(
        '--log_name',
        type=str,
        default='NATSMAETrainer',
        help='name of this experiments')

    parser.add_argument(
        '--classes', type=int, default=10, help='dataset classes')
    parser.add_argument('--layers', type=int, default=20, help='batch size')
    parser.add_argument(
        '--num_choices', type=int, default=4, help='number choices per layer')
    parser.add_argument(
        '--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--epochs', type=int, default=600, help='batch size')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.025,
        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument(
        '--weight-decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument(
        '--val_interval',
        type=int,
        default=5,
        help='validate and save frequency')
    parser.add_argument(
        '--random_search',
        type=int,
        default=1000,
        help='validate and save frequency')
    # ******************************* dataset *******************************#
    parser.add_argument(
        '--dataset', type=str, default='cifar10', help='path to the dataset')
    parser.add_argument('--cutout', action='store_true', help='use cutout')
    parser.add_argument(
        '--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument(
        '--auto_aug',
        action='store_true',
        default=False,
        help='use auto augmentation')
    parser.add_argument(
        '--resize', action='store_true', default=False, help='use resize')
    args = parser.parse_args()
    print(args)
    return args


def main():
    # args & device
    logger = get_logger('mae')

    args = get_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    # dump config files
    if not os.path.exists(cfg.work_dir):
        os.mkdir(cfg.work_dir)
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # dataset
    assert cfg.dataset in ['cifar10', 'imagenet']

    dataloader = build_loader_simmim(logger)

    model = build_model(cfg.model_name)

    mutator = OneShotMutator(custom_group=None)
    mutator.prepare_from_supernet(model)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), cfg.learning_rate,
                                cfg.momentum, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(  # noqa: F841
        optimizer, lambda epoch: 1 - (epoch / cfg.epochs))

    model = model.to(device)

    trainer = build_trainer(
        cfg.trainer_name,
        model=model,
        mutator=None,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        searching=True,
        device=device,
        log_name=cfg.log_name)

    start = time.time()

    trainer.fit(dataloader, dataloader, cfg.epochs)

    utils.time_record(start)


if __name__ == '__main__':
    main()
