import random
import unittest
from unittest import TestCase

import torch
import torch.nn as nn

from nanonas.models.nasbench201.oneshot_nasbench201 import \
    OneShotNASBench201Network
from nanonas.nas.mutators import OneShotMutator
from nanonas.predictor.pruners.measures.te_nas import compute_tenas


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        current_c = random.randint(8, 20)
        self.c1 = nn.Conv2d(3, current_c, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(current_c, 32, kernel_size=1, stride=2, padding=0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = self.c2(self.c1(x))
        x = self.gap(x)
        x = x.view(-1, 32)
        return self.linear(x)

    def forward_before_global_avg_pool(self, x):
        x = self.c1(x)
        x = self.relu(x)
        return self.c2(x)


class TestTenasScore(TestCase):

    def test_tenas_score_with_fixmodel(self):
        inputs = torch.randn(4, 3, 32, 32)
        for _ in range(3):
            m = ToyModel().cuda()
            score = compute_tenas(net=m, inputs=inputs, targets=None)
            assert score is not None

    def test_nb201_tenas_score_with_mutablemodel(self):
        inputs = torch.randn(4, 3, 32, 32)

        m = OneShotNASBench201Network().cuda()
        o = OneShotMutator(with_alias=True)
        o.prepare_from_supernet(m)

        for i in range(10):
            rand_subnet = o.random_subnet
            o.set_subnet(rand_subnet)
            score = compute_tenas(net=m, inputs=inputs, targets=None)

            print(f'The score of {i} th model is {score}')


if __name__ == '__main__':
    unittest.main()
