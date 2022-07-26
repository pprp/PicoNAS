import argparse
import json
import random
from typing import Dict, List

import torch
import torch.nn as nn
import yaml

from pplib.datasets.data_simmim import build_loader_simmim
from pplib.models.nats.nats_supernet import MAESupernetNATS
from pplib.trainer import NATSTrainer
from pplib.trainer.nats_trainer import MAENATSTrainer
from pplib.utils.logging import get_logger
from pplib.utils.misc import convert_channel2idx
from pplib.utils.rank_consistency import kendalltau, pearson, spearman


def load_json(path):
    with open(path, 'r') as f:
        arch_dict = json.load(f)
    return arch_dict


def compuate_rank_consistency(loader, sampled_dict: Dict,
                              trainer: NATSTrainer) -> None:
    """compute rank consistency of different types of indicators."""
    true_indicator_list: List[float] = []
    supernet_indicator_list: List[float] = []

    for i, (k, v) in enumerate(sampled_dict.items()):
        print(f'evaluating the {i}th architecture.')
        current_op_list = convert_channel2idx(k)
        loss = trainer.metric_score(loader, current_op_list=current_op_list)

        supernet_indicator_list.append(loss)
        true_indicator_list.append(v)

    kt = kendalltau(true_indicator_list, supernet_indicator_list)
    ps = pearson(true_indicator_list, supernet_indicator_list)
    sp = spearman(true_indicator_list, supernet_indicator_list)

    print(f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('rank evaluation')
    parser.add_argument(
        '--bench_path',
        type=str,
        default='./data/benchmark/nats_cifar10_acc_rank.yaml',
        help='benchmark file path')

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='checkpoints/nats_mae/nats_mae_macro_ckpt_0101.pth.tar',
        help='path of supernet checkpoint.')

    parser.add_argument(
        '--num_sample',
        type=int,
        default=50,
        help='number of sample for rank evaluation.')

    args = parser.parse_args()

    logger = get_logger('eval_mae')

    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load arch from benchmark files
    with open(args.bench_path, 'r') as f:
        bench = yaml.load(f)

    # random sample `num_sample` archs
    sampled_archs: List[str] = random.sample(bench.keys(), k=args.num_sample)

    # generate sampled dict
    sampled_dict: Dict = {arch: bench[arch] for arch in sampled_archs}

    # generate supernet model
    supernet = MAESupernetNATS('cifar10')

    # criterion
    criterion = nn.MSELoss().to(device)

    # load supernet checkpoints
    state = torch.load(args.ckpt_path)['state_dict']
    supernet.load_state_dict(state, strict=False)

    # build valid dataloader
    dataloader = build_loader_simmim(is_train=False)

    # get trainer
    trainer = MAENATSTrainer(
        supernet, mutator=None, device=device, criterion=criterion)

    # compute the rank consistency of supernet
    compuate_rank_consistency(
        loader=dataloader, sampled_dict=sampled_dict, trainer=trainer)
