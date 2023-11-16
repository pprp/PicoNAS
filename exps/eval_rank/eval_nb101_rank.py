import argparse
import os

import torch
import torch.nn.functional as F
from nasbench import api
from nasbench_pytorch.model import Network as NBNetwork
from tqdm import tqdm

from piconas.datasets.build import build_dataloader
from piconas.evaluator.nb101_evaluator import RANDOM_SAMPLED_HASHES
from piconas.predictor.pruners.measures.zico import getzico
from piconas.predictor.pruners.predictive import find_measures
from piconas.utils.rank_consistency import kendalltau, spearman


def get_args():
    parser = argparse.ArgumentParser('train nb101 benchmark')
    parser.add_argument(
        '--work_dir', type=str, default='./work_dir', help='experiment name'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data/cifar', help='path to the dataset'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='OneShotNASBench101Network',
        help='name of model',
    )
    parser.add_argument(
        '--trainer_name', type=str, default='NB101Trainer', help='name of trainer'
    )
    parser.add_argument(
        '--log_name',
        type=str,
        default='NB101Trainer',
        help='name of this experiments',
    )

    parser.add_argument('--crit', type=str, default='mse', help='decide the criterion')
    parser.add_argument(
        '--optims', type=str, default='sgd', help='decide the optimizer'
    )
    parser.add_argument(
        '--sched', type=str, default='cosine', help='decide the scheduler'
    )

    parser.add_argument('--classes', type=int, default=10, help='dataset classes')
    parser.add_argument('--layers', type=int, default=20, help='batch size')
    parser.add_argument(
        '--num_choices', type=int, default=4, help='number choices per layer'
    )
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='batch size')
    parser.add_argument('--lr', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument(
        '--val_interval', type=int, default=5, help='validate and save frequency'
    )
    parser.add_argument(
        '--random_search', type=int, default=1000, help='validate and save frequency'
    )
    # ******************************* dataset *******************************#
    parser.add_argument(
        '--dataset', type=str, default='cifar10', help='path to the dataset'
    )
    parser.add_argument('--cutout', action='store_true', help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument(
        '--auto_aug', action='store_true', default=False, help='use auto augmentation'
    )
    parser.add_argument(
        '--resize', action='store_true', default=False, help='use resize'
    )
    args = parser.parse_args()
    return args


def main():
    cfg = get_args()

    print(cfg)

    # dump config files
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.trainer_name)
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
    # cfg.dump(os.path.join(cfg.work_dir, current_exp_name))

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataloader = build_dataloader(type='train', dataset=cfg.dataset, config=cfg)

    zc_name_list = ['eznas-a']

    nb101_api = api.NASBench('/data/lujunl/pprp/bench/nasbench_only108.tfrecord')

    for zc_name in zc_name_list:
        p_scores, gt_scores = [], []
        for _hash in tqdm(RANDOM_SAMPLED_HASHES):
            # build network
            ops = nb101_api.get_metrics_from_hash(_hash)[0]['module_operations']
            adj = nb101_api.get_metrics_from_hash(_hash)[0]['module_adjacency']
            net = NBNetwork((adj, ops))

            # score
            if zc_name == 'zico':
                score = getzico(
                    net, train_dataloader, loss_fn=F.cross_entropy, split_data=1
                )
            else:
                score = find_measures(
                    net,
                    train_dataloader,
                    dataload_info=['random', 3, 10],
                    measure_names=[zc_name],
                    loss_fn=F.cross_entropy,
                    device=device,
                )

            gt = nb101_api.get_metrics_from_hash(_hash)[1][108][0][
                'final_test_accuracy'
            ]

            p_scores.append(score)
            gt_scores.append(gt)

        print(f'zc_name: {zc_name}')
        print(f'kendalltau: {kendalltau(p_scores, gt_scores)}')
        print(f'spearman: {spearman(p_scores, gt_scores)}')
        print('===' * 5)


if __name__ == '__main__':
    main()
