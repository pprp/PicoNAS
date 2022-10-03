import random
import unittest
from unittest import TestCase

import torch
import torch.nn as nn

from pplib.datasets import build_dataloader
from pplib.models.nasbench201.oneshot_nasbench201 import \
    OneShotNASBench201Network
from pplib.nas.mutators import OneShotMutator
from pplib.predictor.pruners.predictive import find_measures_arrays
from pplib.predictor.zerocost import ZeroCost


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        current_c = random.randint(8, 20)
        self.c1 = nn.Conv2d(3, current_c, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(current_c, 32, kernel_size=1, stride=2, padding=0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(32, 10)
        self.num_classes = 10

    def forward(self, x):
        x = self.c2(self.c1(x))
        x = self.gap(x)
        x = x.view(-1, 32)
        return self.linear(x)

    def forward_before_global_avg_pool(self, x):
        x = self.c1(x)
        x = self.relu(x)
        return self.c2(x)


class TestPredictive(TestCase):

    def setUp(self) -> None:
        self.model = OneShotNASBench201Network()
        # self.model = ToyModel()
        self.mutator = OneShotMutator(with_alias=True)
        self.mutator.prepare_from_supernet(self.model)
        self.dataloader = build_dataloader('cifar10', 'train')

    # def test_nwot(self):
    #
    #     # dataload, num_imgs_or_batches, num_classes
    #     dataload_info = ['random', 1, 10]
    #     device = torch.device('cuda')
    #     measure_values = find_measures_arrays(net_orig=self.model,
    #                                           trainloader=dataloader,
    #                                           dataload_info=dataload_info,
    #                                           measure_names=None,
    #                                           device=device)

    #     for k, v in measure_values.items():
    #         print(f"k: {k} => {v}")

    def test_zero_cost(self):
        print('==> jacov')
        for _ in range(3):
            predictor = ZeroCost(method_type='jacov')
            score = predictor.query(self.model, self.dataloader)
            print(score)
        print('==> snip')
        for _ in range(3):
            predictor = ZeroCost(method_type='snip')
            score = predictor.query(self.model, self.dataloader)
            print(score)
        print('==> synflow')
        for _ in range(3):
            predictor = ZeroCost(method_type='synflow')
            score = predictor.query(self.model, self.dataloader)
            print(score)
        print('==> grad_norm')
        for _ in range(3):
            predictor = ZeroCost(method_type='grad_norm')
            score = predictor.query(self.model, self.dataloader)
            print(score)
        print('==> fisher')
        for _ in range(3):
            predictor = ZeroCost(method_type='fisher')
            score = predictor.query(self.model, self.dataloader)
            print(score)
        print('==> grasp')
        for _ in range(3):
            predictor = ZeroCost(method_type='grasp')
            score = predictor.query(self.model, self.dataloader)
            print(score)


if __name__ == '__main__':
    unittest.main()
