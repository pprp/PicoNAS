import argparse
import json
import random
from typing import Dict, List

import torch

from pplib.datasets import build_dataloader
from pplib.models import MacroBenchmarkSuperNet
from pplib.nas.mutators import OneShotMutator
from pplib.trainer import MacroTrainer
from pplib.utils.misc import convert_arch2dict
from pplib.utils.rank_consistency import kendalltau, pearson, spearman


class CostumDict(dict):
    __setattr__ = dict.__setitem__
    __getattribute__ = dict.__getitem__


def load_json(path):
    with open(path, 'r') as f:
        arch_dict = json.load(f)
    return arch_dict


def compuate_rank_consistency(sampled_dict: Dict,
                              trainer: MacroTrainer,
                              type: str = 'test_acc') -> None:
    """compute rank consistency of different types of indicators."""
    assert type in ['test_acc', 'MMACs', 'val_acc', 'Params'], \
        f'Not support type {type}.'

    # compute true indicator list
    # [type]: test_acc, MMACs, val_acc, or Params
    true_indicator_list: List[float] = []
    supernet_indicator_list: List[float] = []

    for i, (k, v) in enumerate(sampled_dict.items()):
        print(f'evaluating the {i}th architecture.')
        subnet_dict = convert_arch2dict(k)
        top1 = trainer.valid(epoch=0, subnet_dict=subnet_dict)
        supernet_indicator_list.append(top1)
        true_indicator_list.append(v[type])

    kt = kendalltau(true_indicator_list, supernet_indicator_list)
    ps = pearson(true_indicator_list, supernet_indicator_list)
    sp = spearman(true_indicator_list, supernet_indicator_list)

    print(f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}.")


if __name__ == '__main__':
    args = argparse.ArgumentParser('rank evaluation')
    args.add_argument(
        '--json-path',
        type=str,
        default='./data/benchmark/benchmark_cifar10_dataset.json',
        help='benchmark json file path')
    args.add_argument(
        '--ckpt-path',
        type=str,
        default='checkpoints/path_to_checkpoint.pth.tar',
        help='path of supernet checkpoint.')
    args.add_argument(
        '--type',
        type=str,
        choices=['test_acc', 'MMACs', 'val_acc', 'Params'],
        default='test_acc',
        help='target type to rank.')
    args.add_argument(
        '--num-sample',
        type=int,
        default=100,
        help='number of sample for rank evaluation.')

    args = args.parse_args()

    valid_args = dict(
        name='cifar10',
        bs=64,
        root='./data/cifar',
        fast=False,
        nw=2,
        random_erase=None,
        autoaugmentation=None,
        cutout=None,
    )
    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load arch from json
    arch_dict = load_json(args.json_path)

    # random sample `num_sample` archs
    sampled_archs: List[str] = random.sample(
        arch_dict.keys(), k=args.num_sample)

    # generate sampled dict
    sampled_dict: Dict = {}
    for arch in sampled_archs:
        sampled_dict[arch] = arch_dict[arch]

    # generate supernet model
    supernet = MacroBenchmarkSuperNet()

    # load supernet checkpoints
    state = torch.load(args.ckpt_path)['state_dict']
    supernet.load_state_dict(state, strict=False)

    # build one-shot mutator
    mutator = OneShotMutator()
    mutator.prepare_from_supernet(supernet)

    # build valid dataloader
    dataloader = {}
    dataloader['val'] = build_dataloader(args=CostumDict(valid_args))

    # get trainer
    trainer = MacroTrainer(
        supernet, mutator=mutator, dataloader=dataloader, device=device)

    # compute the rank consistency of supernet
    compuate_rank_consistency(
        sampled_dict=sampled_dict, trainer=trainer, type=args.type)
