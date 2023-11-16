import argparse
import json
import random
from typing import Dict, List

import torch

from piconas.datasets import build_dataloader
from piconas.models import MacroBenchmarkSuperNet
from piconas.nas.mutators import OneShotMutator
from piconas.trainer import MacroTrainer
from piconas.utils.misc import convert_arch2dict
from piconas.utils.rank_consistency import kendalltau, pearson, spearman


def load_json(path):
    with open(path, 'r') as f:
        arch_dict = json.load(f)
    return arch_dict


def compuate_rank_consistency(
    loader, sampled_dict: Dict, trainer: MacroTrainer, type: str = 'val_acc'
) -> None:
    """compute rank consistency of different types of indicators."""
    assert type in [
        'test_acc',
        'MMACs',
        'val_acc',
        'Params',
    ], f'Not support type {type}.'

    # compute true indicator list
    # [type]: test_acc, MMACs, val_acc, or Params
    true_indicator_list: List[float] = []
    supernet_indicator_list: List[float] = []

    for i, (k, v) in enumerate(sampled_dict.items()):
        print(f'evaluating the {i}th architecture.')
        subnet_dict = convert_arch2dict(k)
        loss, top1_acc, top5_acc = trainer.metric_score(
            loader, subnet_dict=subnet_dict)

        supernet_indicator_list.append(top1_acc)
        true_indicator_list.append(v[type])

    kt = kendalltau(true_indicator_list, supernet_indicator_list)
    ps = pearson(true_indicator_list, supernet_indicator_list)
    sp = spearman(true_indicator_list, supernet_indicator_list)

    print(f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('rank evaluation')
    parser.add_argument(
        '--json_path',
        type=str,
        default='./data/benchmark/benchmark_cifar10_dataset.json',
        help='benchmark json file path',
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='checkpoints/path_to_checkpoint.pth.tar',
        help='path of supernet checkpoint.',
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['test_acc', 'MMACs', 'val_acc', 'Params'],
        default='test_acc',
        help='target type to rank.',
    )
    parser.add_argument(
        '--num_sample',
        type=int,
        default=100,
        help='number of sample for rank evaluation.',
    )

    cfg = parser.parse_args()

    valid_args = dict(
        name='cifar10',
        bs=64,
        data_dir='./data/cifar',
        fast=False,
        nw=2,
        random_erase=False,
        autoaugmentation=False,
        cutout=False,
        batch_size=32,
    )

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load arch from json
    arch_dict = load_json(cfg.json_path)

    # random sample `num_sample` archs
    sampled_archs: List[str] = random.sample(
        arch_dict.keys(), k=cfg.num_sample)

    # generate sampled dict
    sampled_dict: Dict = {arch: arch_dict[arch] for arch in sampled_archs}
    # generate supernet model
    supernet = MacroBenchmarkSuperNet()

    # load supernet checkpoints
    state = torch.load(cfg.ckpt_path)['state_dict']
    supernet.load_state_dict(state, strict=False)

    # build one-shot mutator
    mutator = OneShotMutator()
    mutator.prepare_from_supernet(supernet)

    # build valid dataloader
    dataloader = build_dataloader(config=cfg, type='val')

    # get trainer
    trainer = MacroTrainer(supernet, mutator=mutator, device=device)

    # compute the rank consistency of supernet
    compuate_rank_consistency(
        loader=dataloader, sampled_dict=sampled_dict, trainer=trainer, type=cfg.type
    )
