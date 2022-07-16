import unittest
from unittest import TestCase

from pplib.datasets import build_dataloader
from pplib.nas.search_spaces import get_search_space
from pplib.utils.config import Config
from pplib.utils.get_dataset_api import get_zc_benchmark_api


class TestZenScore(TestCase):

    def test_zen_score(self):
        args = dict(
            batch_size=64,
            fast=False,
            nw=2,
            random_erase=False,
            autoaugmentation=None,
            cutout=None,
            data_dir='./data/cifar')
        dataloader = build_dataloader(config=Config(args))

        search_space = get_search_space('nasbench201', 'cifar10')
        zc_api = get_zc_benchmark_api('nasbench201', 'cifar10')  # dict
        search_space.instantiate_model = False
        search_space.labeled_archs = [eval(arch) for arch in zc_api.keys()]

        method_type = 'grad_norm'
        loss_fn = search_space.get_loss_fn()


if __name__ == '__main__':
    unittest.main()
