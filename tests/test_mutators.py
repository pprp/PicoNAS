from unittest import TestCase

import torch.nn as nn
from torch import Tensor

from pplib.nas.mutables import OneShotOP
from pplib.nas.mutators import OneShotMutator


class OneShotMutableModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.op1 = OneShotOP(
            candidate_ops=nn.ModuleDict({
                'cand1': nn.Conv2d(32, 32, 3, 1, 1),
                'cand2': nn.Conv2d(32, 32, 5, 1, 2),
                'cand3': nn.Conv2d(32, 32, 7, 1, 3),
            }))
        self.op2 = OneShotOP(
            candidate_ops=nn.ModuleDict({
                'cand1': nn.Conv2d(32, 32, 3, 1, 1),
                'cand2': nn.Conv2d(32, 32, 5, 1, 2),
                'cand3': nn.Conv2d(32, 32, 7, 1, 3),
            }))
        self.op3 = OneShotOP(
            candidate_ops=nn.ModuleDict({
                'cand1': nn.Conv2d(32, 32, 3, 1, 1),
                'cand2': nn.Conv2d(32, 32, 5, 1, 2),
                'cand3': nn.Conv2d(32, 32, 7, 1, 3),
            }))

    def forward(self, x: Tensor) -> Tensor:
        x = self.op1(x)
        x = self.op2(x)
        return self.op3(x)


class TestOneShot(TestCase):

    def test_case1(self):
        supernet = OneShotMutableModel()

        custom_group = [['op1', 'op2', 'op3']]

        mutator = OneShotMutator(custom_group=custom_group)
        mutator.prepare_from_supernet(supernet)
        print(mutator.search_group)

        print(mutator.random_subnet)

    def test_case2(self):
        supernet = OneShotMutableModel()

        custom_group = [['op1'], ['op2'], ['op3']]

        mutator = OneShotMutator(custom_group=custom_group)
        mutator.prepare_from_supernet(supernet)
        print(mutator.search_group)
        print(mutator.random_subnet)


if __name__ == '__main__':
    import unittest
    unittest.main()
