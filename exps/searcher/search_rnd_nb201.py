import argparse
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import piconas.utils.utils as utils
from piconas.core import build_criterion, build_optimizer, build_scheduler
from piconas.datasets.build import build_dataloader
from piconas.evaluator.nb201_evaluator import NB201Evaluator
from piconas.models import build_model
from piconas.predictor.pruners.predictive import find_measures
from piconas.searcher import RandomSearcher
from piconas.trainer import build_trainer
from piconas.utils import set_random_seed


def get_args():
    parser = argparse.ArgumentParser('train macro benchmark')
    parser.add_argument(
        '--work_dir',
        type=str,
        default='./logdir/rnd_search',
        help='experiment name')

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
    # measure_name
    parser.add_argument(
        '--measure_name',
        type=str,
        default='eznas-a',
        help='measure name of zerocost proxies')
    # is_predictor
    parser.add_argument(
        '--is_predictor',
        type=bool,
        default=False,
        help='whether to use predictor')

    return parser.parse_args()


def main():
    args = get_args()

    # set envirs
    set_random_seed(args.seed, deterministic=True)

    print(args)

    # dump config files
    args.work_dir = os.path.join(args.work_dir, args.trainer_name)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataloader = build_dataloader(
        type='train', dataset=args.dataset, config=args)

    val_dataloader = build_dataloader(
        type='val', dataset=args.dataset, config=args)

    # build model
    model = build_model(args.model_name)
    model = model.to(device)

    criterion = build_criterion(args.crit).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(args, optimizer)

    # build trainer
    trainer = build_trainer(
        args.trainer_name,
        model=model,
        mutator=None,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        searching=True,
        device=device,
        log_name=args.log_name,
    )

    # Random Search Algorithm for Zero-cost proxies

    # Build trainer and evaluator
    evaluator = NB201Evaluator(
        trainer, num_sample=50, is_predictor=args.is_predictor)

    # Record the best one with highest zc score
    best_subnet = None
    best_zc_score = -1
    corresponding_gt_score = None

    # Record the whole process for visualization
    trial_list, best_zc_score_list, best_gt_score_list = [], [], []

    # Directly implement random search
    for i in tqdm(range(1000)):
        # random sample one subnet config
        rnd_subnet = trainer.mutator.random_subnet
        # set config to model
        trainer.mutator.set_subnet(rnd_subnet)
        # generate genotype
        genotype = evaluator.generate_genotype(rnd_subnet, trainer.mutator)
        # query results
        gt_score = evaluator.query_result(genotype)  # type is eval_acc1es

        # set parameters for measure zc
        dataload_info = ['random', 3, 10]  # class number
        # get the zc score
        zc_score = find_measures(
            trainer.model,
            train_dataloader,
            dataload_info=dataload_info,
            measure_names=args.measure_name if isinstance(
                args.measure_name, list) else [args.measure_name],
            loss_fn=F.cross_entropy,
            device=device)
        # update the best record
        if zc_score > best_zc_score:
            best_zc_score = zc_score
            best_subnet = rnd_subnet
            corresponding_gt_score = gt_score
            print(f'Best Subnet Found: {best_subnet}')
            print(f'Best Zc Score of {args.measure_name} is {best_zc_score}')
            print(f'Corresponding gt score is {corresponding_gt_score}')

        # record the whole process
        trial_list.append(i)
        best_zc_score_list.append(best_zc_score)
        best_gt_score_list.append(corresponding_gt_score)

    # Print the final results
    print('===' * 5)
    print(f'Best Subnet Found: {best_subnet}')
    print(f'Best Zc Score of {args.measure_name} is {best_zc_score}')
    print(f'Corresponding gt score is {corresponding_gt_score}')
    print('===' * 5)

    # save the record to .csv file with path=./logdir/evo_search
    import pandas as pd
    _dict = {
        'trials': trial_list,
        'zc_score': best_zc_score_list,
        'gt_score': best_gt_score_list
    }
    df = pd.DataFrame(_dict)
    df.to_csv(os.path.join(args.work_dir, f'{args.log_name}.csv'))

    # save ploted figure to .png file with path=./logdir/evo_search
    import matplotlib.pyplot as plt
    plt.plot(trial_list, best_zc_score_list, label='zc_score')
    plt.plot(trial_list, best_gt_score_list, label='gt_score')
    plt.legend()
    plt.xlabel('trials')
    plt.ylabel('score')
    plt.savefig(os.path.join(args.work_dir, f'{args.log_name}.png'))


if __name__ == '__main__':
    main()
