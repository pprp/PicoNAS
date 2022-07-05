import unittest 
from unittest import TestCase
import torch

from pplib.datasets import build_dataloader
from pplib.nas.search_spaces import get_search_space
from pplib.nas.search_spaces.core.query_metrics import Metric
from pplib.predictor import predictive
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
        zc_api = get_zc_benchmark_api('nasbench201', 'cifar10') # dict 
        search_space.instantiate_model = False
        search_space.labeled_archs = [eval(arch) for arch in zc_api.keys()]

        method_type = 'grad_norm'
        loss_fn = search_space.get_loss_fn()

        search_space.query(
            metric=Metric.VAL_ACCURACY, dataset="cifar10",
            dataset_api=zc_api,
        )

        score = predictive.find_measures(
            net_orig=search_space,
            dataloader=dataloader,
            dataload_info=('random', 1, 10),
            device=torch.device('cpu'),
            loss_fn=loss_fn,
            measure_names=[method_type],
        )

        print(score)


if __name__ == '__main__':
    unittest.main()
