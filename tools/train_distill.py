import argparse
import os
import time

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import nanonas.utils.utils as utils
from nanonas.core import build_optimizer, build_scheduler
from nanonas.core.losses import build_criterion
from nanonas.datasets.transforms.cutout import Cutout
from nanonas.models import resnet20, resnet56
from nanonas.trainer import Distill_Trainer
from nanonas.utils import set_random_seed
from nanonas.utils.config import Config


def get_args():
    parser = argparse.ArgumentParser('train macro benchmark')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/spos/spos_cifar10.py',
        required=False,
        help='user settings config',
    )
    parser.add_argument(
        '--work_dir', type=str, default='./work_dir', help='experiment name')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/cifar',
        help='path to the dataset')

    parser.add_argument(
        '--seed', type=int, default=42, help='seed of experiments')

    parser.add_argument(
        '--model_name',
        type=str,
        default='DiffNASBench201Network',
        help='name of model')
    parser.add_argument(
        '--trainer_name',
        type=str,
        default='Darts_Trainer',
        help='name of trainer')
    parser.add_argument(
        '--log_name',
        type=str,
        default='test_darts_Darts_Trainer-epoch50',
        help='name of this experiments',
    )

    # ******************************* settings *******************************#

    parser.add_argument(
        '--crit', type=str, default='ce', help='decide the criterion')
    parser.add_argument(
        '--optims', type=str, default='sgd', help='decide the optimizer')
    parser.add_argument(
        '--sched', type=str, default='cosine', help='decide the scheduler')

    parser.add_argument(
        '--classes', type=int, default=10, help='dataset classes')
    parser.add_argument('--layers', type=int, default=20, help='batch size')
    parser.add_argument(
        '--num_choices', type=int, default=4, help='number choices per layer')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument(
        '--weight-decay', type=float, default=5e-4, help='weight decay')
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
        '--dataset', type=str, default='simmim', help='path to the dataset')
    parser.add_argument('--cutout', action='store_true', help='use cutout')
    parser.add_argument(
        '--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument(
        '--auto_aug',
        action='store_true',
        default=False,
        help='use auto augmentation')
    parser.add_argument(
        '--train_portion',
        type=float,
        default=0.5,
        help='portion of training data')

    parser.add_argument(
        '--resize', action='store_true', default=False, help='use resize')
    return parser.parse_args()


def data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(length=args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def main():
    args = get_args()

    # merge argparse and config file
    if args.config is not None:
        cfg = Config.fromfile(args.config)
        cfg.merge_from_dict(vars(args))
    else:
        cfg = Config(args)

    # set envirs
    set_random_seed(cfg.seed, deterministic=True)

    print(cfg)

    # dump config files
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.trainer_name)
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
    current_exp_name = f'{cfg.model_name}-{cfg.trainer_name}-{cfg.log_name}.yaml'
    cfg.dump(os.path.join(cfg.work_dir, current_exp_name))

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_transform, valid_transform = data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=2)

    val_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True,
        num_workers=2)

    model_s = resnet20()
    model_t = resnet56()

    # TODO load teacher model

    criterion = build_criterion(cfg.crit)
    optimizer = build_optimizer(model_s, cfg)
    scheduler = build_scheduler(cfg, optimizer)

    model_s = model_s.to(device)
    model_t = model_t.to(device)

    trainer = Distill_Trainer(
        model=model_s,
        teacher=model_t,
        mutator=None,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        searching=True,
        device=device,
        log_name=cfg.log_name,
    )

    start = time.time()

    trainer.fit(train_loader, val_loader, cfg.epochs)

    utils.time_record(start)


if __name__ == '__main__':
    main()
