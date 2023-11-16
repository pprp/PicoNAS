import unittest
from unittest import TestCase

import torch
import torch.nn as nn

from piconas.nas.mutables import MutableValue
from piconas.nas.mutators import ValueMutator


class ToyBlock(nn.Module):
    """
    depth_list=[3,5,7],
    kernel_list=[3, 5],
    channel_list=[0.5, 1]
    """

    def __init__(self, depth=max([3, 5, 7]), kernel=2, channel=0.5):
        super().__init__()
        self.mutable_depth = MutableValue(
            value_list=[3, 5, 7], default_value=7)
        self.m = nn.ModuleList()
        for _ in range(depth):
            self.m.append(
                nn.Conv2d(
                    int(64 * channel),
                    int(64 * channel),
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                )
            )

    def forward(self, x):
        current_depth = self.mutable_depth.current_choice
        for i in range(current_depth):
            x = self.m[i](x)
        return x


class TestMutableValue(TestCase):
    def test_mutable_value(self):
        m = ToyBlock()
        i = torch.randn(1, 32, 32, 32)
        o = m(i)

    def test_value_mutator(self):
        m = ToyBlock()
        v = ValueMutator()
        v.prepare_from_supernet(m)
        print(v.search_group)


if __name__ == '__main__':
    unittest.main()
