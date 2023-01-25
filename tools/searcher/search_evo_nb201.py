import argparse
import os
import time

import torch

import nanonas.utils.utils as utils
from nanonas.core import build_criterion, build_optimizer, build_scheduler
from nanonas.datasets.build import build_dataloader
from nanonas.models import build_model
from nanonas.searcher import EvolutionSearcher
from nanonas.trainer import build_trainer
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
        '--model_path', type=str, default='', help='model path')

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
        default='OneShotNASBench201Network',
        help='name of model')
    parser.add_argument(
        '--trainer_name',
        type=str,
        default='NB201Trainer',
        help='name of trainer')
    parser.add_argument(
        '--log_name',
        type=str,
        default='NB201Trainer',
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
        '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='batch size')
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
    return parser.parse_args()


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

    train_dataloader = build_dataloader(
        type='train', dataset=cfg.dataset, config=cfg)

    val_dataloader = build_dataloader(
        type='val', dataset=cfg.dataset, config=cfg)

    if cfg.dataset == 'cifar10':
        num_classes = 10
    elif cfg.dataset == 'cifar100':
        num_classes = 100
    elif cfg.dataset == 'ImageNet16-120':
        num_classes = 120
    else:
        raise NotImplementedError(
            f'Not Support Type of datasets: {cfg.dataset}.')

    model = build_model(cfg.model_name, num_classes=num_classes)

    criterion = build_criterion(cfg.crit).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg, optimizer)

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
        log_name=cfg.log_name,
        dataset=cfg.dataset,
    )

    start = time.time()
    print('Begin to search....')

    # set model path
    searcher = EvolutionSearcher(
        max_epochs=20,
        select_num=10,
        population_num=50,
        crossover_num=25,
        mutation_num=25,
        trainer=trainer,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        model_path=cfg.model_path,
        log_name=cfg.log_name,
        logger=trainer.logger)

    searcher.search()

    utils.time_record(start)


if __name__ == '__main__':
    main()
