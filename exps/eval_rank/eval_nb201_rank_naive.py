import argparse
import os
import random

import torch
from tqdm import tqdm 
from nas_201_api import NASBench201API

from piconas.datasets.build import build_dataloader
from piconas.models.nasbench201.apis.utils import dict2config, get_cell_based_tiny_net
from piconas.predictor.pruners.predictive import find_measures
import torch.nn.functional as F
from piconas.utils.rank_consistency import kendalltau, spearman, pearson


nb201_api = NASBench201API(
    file_path_or_dict='data/NAS-Bench-201-v1_1-096897.pth', verbose=False)

def random_sample_and_get_gt():
    index_range = list(range(15625))
    choiced_index = random.choice(index_range)
    # modelinfo is a index

    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': 10
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(choiced_index, dataset='cifar10', hp='200')
    return choiced_index, model, xinfo['test-accuracy']

def index_sample_and_get_gt(choiced_index):
    assert choiced_index < 15625

    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': 10
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    xinfo = nb201_api.get_more_info(choiced_index, dataset='cifar10', hp='200')
    return choiced_index, model, xinfo['test-accuracy']

def get_args():
    parser = argparse.ArgumentParser('train nb201 benchmark')
    parser.add_argument(
        '--work_dir', type=str, default='./work_dir', help='experiment name'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data/cifar', help='path to the dataset'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='OneShotNASBench201Network',
        help='name of model',
    )
    parser.add_argument(
        '--trainer_name', type=str, default='NB201Trainer', help='name of trainer'
    )
    parser.add_argument(
        '--log_name',
        type=str,
        default='NB201Trainer',
        help='name of this experiments',
    )

    parser.add_argument('--crit', type=str, default='mse',
                        help='decide the criterion')
    parser.add_argument(
        '--optims', type=str, default='sgd', help='decide the optimizer'
    )
    parser.add_argument(
        '--sched', type=str, default='cosine', help='decide the scheduler'
    )

    parser.add_argument('--classes', type=int, default=10,
                        help='dataset classes')
    parser.add_argument('--layers', type=int, default=20, help='batch size')
    parser.add_argument(
        '--num_choices', type=int, default=4, help='number choices per layer'
    )
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='batch size')
    parser.add_argument('--lr', type=float, default=0.025,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float,
                        default=5e-4, help='weight decay')
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
    parser.add_argument('--cutout_length', type=int,
                        default=16, help='cutout length')
    parser.add_argument(
        '--auto_aug', action='store_true', default=False, help='use auto augmentation'
    )
    parser.add_argument(
        '--resize', action='store_true', default=False, help='use resize'
    )
    args = parser.parse_args()
    return args


def evaluate_ranking(train_dataloader, device, zc_name_list, num_samples=1000):
    # for CIFAR10
    
    dataload_info = ['random', 3, 10]
    
    for zc_name in zc_name_list:
        zc_list, gt_list = [], []
        for i in tqdm(range(num_samples)):
            # _, model, gt = random_sample_and_get_gt()
            _, model, gt = index_sample_and_get_gt(i)
            zc = find_measures(
                    model,
                    train_dataloader,
                    dataload_info=dataload_info,
                    measure_names=[zc_name],
                    loss_fn=F.cross_entropy,
                    device=device,
                )
            zc_list.append(zc)
            gt_list.append(gt)

        print(f'ZC: {zc_name}')
        print(f'Kendalltau: {kendalltau(zc_list, gt_list)}')
        print(f'Spearman: {spearman(zc_list, gt_list)}')
        print(f'Pearson: {pearson(zc_list, gt_list)}')
        print('-----------------------------------')

def main():
    cfg = get_args()

    print(cfg)

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

    num_samples = 50
    # max is 15625
    
    zc_name_list = [
        'grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'l2_norm',
    ]
    
    evaluate_ranking(train_dataloader, device, zc_name_list=zc_name_list, num_samples=num_samples)


if __name__ == '__main__':
    main()
