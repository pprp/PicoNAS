import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from piconas.datasets.build import build_dataloader
from piconas.predictor.pruners.measures.zico import getzico
from piconas.predictor.pruners.predictive import find_measures
from piconas.utils.get_dataset_api import NDS
from piconas.utils.rank_consistency import kendalltau, spearman

BASE = './checkpoints/nds_data/'


def get_args():
    parser = argparse.ArgumentParser('train nb201 benchmark')
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
        default='NB201Trainer',
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
        '--batch_size', type=int, default=64, help='batch size')
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
    parser.add_argument('--search_space', type=str, default='Amoeba')
    # --gpu_id $GPU
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    cfg = get_args()

    print(cfg)

    search_space_list = ['Amoeba', 'DARTS', 'ENAS', 'NASNet', 'PNAS']

    assert cfg.search_space in search_space_list, 'search space is invalid'

    # set torch devices based on gpu_id
    torch.cuda.set_device(cfg.gpu_id)

    # dump config files
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.trainer_name)
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataloader = build_dataloader(
        type='train', dataset=cfg.dataset, config=cfg)

    dataload_info = ['random', 3, 10]

    nds_api = NDS(cfg.search_space)

    # zc_name_list = ['fisher', 'grad_norm', 'grasp', 'l2_norm', 'plain', 'snip', 'synflow', 'epe_nas', 'jacov', 'nwot', 'zen', 'zico', 'eznas-a', 'flops', 'params']
    # zc_name_list = ['epe_nas', 'jacov', 'zen', 'zico', 'eznas-a']
    zc_name_list = ['eznas-a']

    for zc_name in zc_name_list:
        p_scores, gt_scores = [], []
        for uid in tqdm(range(100)):
            if True:
                model = nds_api[uid]

                if zc_name == 'zico':
                    score = getzico(
                        model,
                        train_dataloader,
                        loss_fn=F.cross_entropy,
                        split_data=1)
                else:
                    score = find_measures(
                        model,
                        train_dataloader,
                        dataload_info=dataload_info,
                        measure_names=[zc_name],
                        loss_fn=F.cross_entropy,
                        device=device)
                gt = nds_api.get_final_accuracy(uid)
                p_scores.append(score)
                gt_scores.append(gt)

        # compute kendall spearman
        kd = kendalltau(p_scores, gt_scores)
        sp = spearman(p_scores, gt_scores)
        print(f'zc_name: {zc_name}')
        print(f'kendalltau: {kd}')
        print(f'spearman: {sp}')
        print('===' * 5)


if __name__ == '__main__':
    main()
