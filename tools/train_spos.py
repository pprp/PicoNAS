import argparse
import os
import time

import init_paths  # noqa: F401
import torch
import torch.nn as nn
from thop import profile

import pplib.utils.utils as utils
from pplib.datasets import build_dataloader
from pplib.models import SearchableShuffleNetV2
from pplib.nas.mutators.one_shot_mutator import OneShotMutator
from pplib.trainer import SPOSTrainer
from pplib.utils.config import Config

# from torchsummary import summary


def parse_args():
    parser = argparse.ArgumentParser('Single_Path_One_Shot')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/spos/spos_cifar10.py',
        required=True,
        help='user settings config')
    parser.add_argument(
        '--exp_name', type=str, default='spos_cifar10', help='experiment name')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/cifar',
        help='path to the dataset')
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
    args = parse_args()

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
    assert cfg.dataset_type in ['cifar10', 'imagenet']
    train_loader = build_dataloader(name='cifar10', type='train', config=cfg)
    val_loader = build_dataloader(name='cifar10', type='val', config=cfg)

    dataloader = {
        'train': train_loader,
        'val': val_loader,
    }

    # SinglePath_OneShot
    model = SearchableShuffleNetV2()
    mutator = OneShotMutator()
    mutator.prepare_from_supernet(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), cfg.learning_rate,
                                cfg.momentum, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - (epoch / cfg.epochs))

    # flops & params & structure
    flops, params = profile(
        model,
        inputs=(torch.randn(1, 3, 32, 32), )
        if cfg.dataset_type == 'cifar10' else (torch.randn(1, 3, 224, 224), ),
        verbose=False)

    print('Random Path of the Supernet: Params: %.2fM, Flops:%.2fM' %
          ((params / 1e6), (flops / 1e6)))
    model = model.to(device)

    trainer = SPOSTrainer(
        model,
        mutator=mutator,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        searching=True,
        epochs=cfg.epochs,
        device=device)

    start = time.time()

    for epoch in range(cfg.epochs):
        trainer.train(epoch)
        if (epoch + 1) % cfg.val_interval == 0:
            trainer.valid(epoch=epoch)
            utils.save_checkpoint({
                'state_dict': model.state_dict(),
            },
                                  epoch + 1,
                                  tag=cfg.exp_name + '_super')

    utils.time_record(start)


if __name__ == '__main__':
    main()
