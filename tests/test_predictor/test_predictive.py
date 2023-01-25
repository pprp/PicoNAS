import random
import unittest
from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanonas.datasets import build_dataloader
from nanonas.models.nasbench201.oneshot_nasbench201 import \
    OneShotNASBench201Network
from nanonas.nas.mutators import OneShotMutator
from nanonas.predictor.pruners.predictive import find_measures


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

    def test_nwot(self):
        """
        'epe_nas' , 'fisher', 'grad_norm', 'grasp' , 'jacov'
        'l2_norm' , 'nwot' , 'plain' , 'snip' , 'synflow', 'zen'
        """
        # dataload, num_imgs_or_batches, num_classes
        dataload_info = ['random', 1, 10]
        device = torch.device('cuda')

        rand_subnet = self.mutator.random_subnet
        self.mutator.set_subnet(rand_subnet)

        measure_values = find_measures(
            net_orig=self.model,
            dataloader=self.dataloader,
            dataload_info=dataload_info,
            measure_names=['zen'],
            loss_fn=F.cross_entropy,
            device=device)

        print(measure_values)


if __name__ == '__main__':
    unittest.main()
