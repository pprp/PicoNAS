import unittest
import warnings
from unittest import TestCase

import torch

from pplib.nas.mutables import DynamicLinear
from pplib.nas.mutables.mutable_value import MutableValue
from pplib.nas.mutators.dynamic_mutator import DynamicMutator


class TestDynamicLinear(TestCase):

    def setUp(self) -> None:
        warnings.simplefilter('ignore', ResourceWarning)

        in_features = MutableValue([4, 6, 8])
        out_features = MutableValue([5, 7, 9])
        self.model = DynamicLinear(in_features, out_features, bias=True)
        self.mutator = DynamicMutator()
        self.mutator.prepare_from_supernet(self.model)

    def test_dynamic_linear(self):
        # test forward during searching
        self.mutator.sample_value(mode='max')
        ins = torch.randn(4, 8)
        outs = self.model(ins)
        print(outs.shape)


if __name__ == '__main__':
    unittest.main()
