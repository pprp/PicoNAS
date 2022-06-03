import argparse
import os
import time

import init_paths  # noqa: F401
import torch
import torch.nn as nn
import torchvision
from thop import profile
# from torchsummary import summary
from torchvision import datasets

import pplib.utils.utils as utils
from pplib.models import SinglePathOneShotSuperNet
from pplib.trainer import SPOSTrainer
from pplib.utils.utils import data_transforms


def get_args():
    parser = argparse.ArgumentParser('Single_Path_One_Shot')
    parser.add_argument(
        '--exp_name',
        type=str,
        default='spos_cifar10',
        required=True,
        help='experiment name')
    parser.add_argument(
        '--data_dir', type=str, default='./data/', help='path to the dataset')
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
    args = get_args()
    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # dataset
    assert args.dataset in ['cifar10', 'imagenet']
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.data_dir, 'cifar'),
            train=True,
            download=True,
            transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8)
        valset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.data_dir, 'cifar'),
            train=False,
            download=True,
            transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8)
    elif args.dataset == 'imagenet':
        train_data_set = datasets.ImageNet(
            os.path.join(args.data_dir, 'ILSVRC2012', 'train'),
            train_transform)
        val_data_set = datasets.ImageNet(
            os.path.join(args.data_dir, 'ILSVRC2012', 'valid'),
            valid_transform)
        train_loader = torch.utils.data.DataLoader(
            train_data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            sampler=None)
        val_loader = torch.utils.data.DataLoader(
            val_data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True)

    dataloader = {
        'train': train_loader,
        'val': val_loader,
    }

    # SinglePath_OneShot
    model = SinglePathOneShotSuperNet(args.dataset, args.resize, args.classes,
                                      args.layers)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - (epoch / args.epochs))

    # flops & params & structure
    flops, params = profile(
        model,
        inputs=(torch.randn(1, 3, 32, 32), ) if args.dataset == 'cifar10' else
        (torch.randn(1, 3, 224, 224), ),
        verbose=False)

    print('Random Path of the Supernet: Params: %.2fM, Flops:%.2fM' %
          ((params / 1e6), (flops / 1e6)))
    model = model.to(device)

    trainer = SPOSTrainer(
        model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        searching=True)

    start = time.time()

    for epoch in range(args.epoch):
        trainer.train(epoch)

        if (epoch + 1) % args.val_interval == 0:
            trainer.valid(epoch)

    utils.time_record(start)


if __name__ == '__main__':
    main()
