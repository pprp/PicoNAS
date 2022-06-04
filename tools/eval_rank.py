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
    """compute rank consistency of different types of indicators.

    Args:
        sampled_dict (Dict): _description_
        supernet (nn.Module): _description_
        type (str): _description_
    """
    assert type in ['test_acc', 'MMACs', 'val_acc', 'Params'], \
        f'Not support type {type}.'

    # compute true indicator list
    # [type]: test_acc, MMACs, val_acc, or Params
    true_indicator_list: List[float] = []
    supernet_indicator_list: List[float] = []

    for k, v in sampled_dict.items():
        subnet_dict = convert_arch2dict(k)
        top1 = trainer.valid(epoch=0, subnet_dict=subnet_dict)
        supernet_indicator_list.append(top1)
        true_indicator_list.append(v[type])

    kt = kendalltau(true_indicator_list, supernet_indicator_list)
    ps = pearson(true_indicator_list, supernet_indicator_list)
    sp = spearman(true_indicator_list, supernet_indicator_list)

    print(f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}.")


if __name__ == '__main__':
    json_path = r'./data/benchmark/benchmark_cifar10_dataset.json'
    ckpt_path = r'checkpoints/log_spos_c10_train_supernet_retrain_super_ckpt_0005.pth.tar'  # noqa: E501
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
    arch_dict = load_json(json_path)

    # random sample 100 archs
    sampled_archs: List[str] = random.sample(arch_dict.keys(), k=100)

    # generate sampled dict
    sampled_dict: Dict = {}
    for arch in sampled_archs:
        sampled_dict[arch] = arch_dict[arch]

    # generate supernet model
    supernet = MacroBenchmarkSuperNet()

    # load supernet checkpoints
    state = torch.load(ckpt_path)['state_dict']
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
    compuate_rank_consistency(sampled_dict=sampled_dict, trainer=trainer)
