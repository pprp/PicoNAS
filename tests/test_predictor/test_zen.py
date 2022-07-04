import pytest
import torch

from pplib.datasets import build_dataloader
from pplib.nas.search_spaces import get_search_space
from pplib.predictor import predictive
from pplib.utils.config import Config


def test_zen_score():
    args = dict(
        batch_size=64,
        fast=False,
        nw=2,
        random_erase=False,
        autoaugmentation=None,
        cutout=None,
        data_dir='./data/cifar')
    dataloader = build_dataloader(config=Config(args))
    search_spaces = get_search_space('nasbench201', 'cifar10')

    method_type = 'grad_norm'
    loss_fn = search_spaces.get_loss_fn()
    score = predictive.find_measures(
        net_orig=search_spaces,
        dataloader=dataloader,
        dataload_info=('random', 1, 10),
        device=torch.device('cuda:0'),
        loss_fn=loss_fn,
        measure_names=[method_type],
    )

    print(score)


if __name__ == '__main__':
    pytest.main()
