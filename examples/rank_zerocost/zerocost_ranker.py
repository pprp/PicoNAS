import argparse
import os
import time

import torch

import nanonas.utils.utils as utils
from nanonas.core import build_criterion, build_optimizer, build_scheduler
from nanonas.evaluator import NB201Evaluator  # noqa: F401
from nanonas.models import build_model
from nanonas.trainer import build_trainer
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
        default='zero_cost_ranker',
        help='name of this experiments',
    )

    parser.add_argument(
        '--crit', type=str, default='mse', help='decide the criterion')
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
        '--batch_size', type=int, default=256, help='batch size')
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


def calculate_zerocost(num_samples: list, trainer, measure_name=None) -> None:
    results = []
    if measure_name is None:
        measure_name = ['flops']

    for num_sample in num_samples:
        evaluator = NB201Evaluator(trainer=trainer, num_sample=num_sample)
        kt, ps, sp, rd_list, minn_at_ks, patks, cpr = evaluator.compute_rank_by_predictive(
            measure_name=measure_name)
        results.append([kt, ps, sp, cpr])

    trainer.logger.info(f'current measure name is: {measure_name[0]}')
    trainer.logger.info(
        "= Num of Samples == Kendall's Tau ==  Pearson   == Spearman == CPR  == "
    )
    for num, result in zip(num_samples, results):
        trainer.logger.info(
            f'=  {num:<6}  \t ==   {result[0]:.4f} \t  ==  {result[1]:.4f} \t ==  {result[2]:.4f} \t == {result[3]:.4f} ='
        )
    trainer.logger.info('=' * 60)


def main():
    args = get_args()

    # merge argparse and config file
    if args.config is not None:
        cfg = Config.fromfile(args.config)
        cfg.merge_from_dict(vars(args))
    else:
        cfg = Config(args)

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

    model = build_model(cfg.model_name, with_residual=True)
    # zenscore: False
    # other: True

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
    )

    start = time.time()

    num_samples = [200]

    for key in [
            'epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
            'nwot', 'plain', 'snip', 'synflow', 'flops', 'params'
    ]:
        calculate_zerocost(num_samples, trainer, measure_name=[key])

    # Note: set `with_residual` to False when testing zenscore
    # calculate_zerocost(num_samples, trainer, measure_name=['zen'])

    utils.time_record(start)


if __name__ == '__main__':
    main()
