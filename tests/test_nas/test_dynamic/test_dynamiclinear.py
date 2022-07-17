import unittest
import warnings
from unittest import TestCase

import pytest
import torch

from pplib.nas.mutables import DynamicLinear
from pplib.nas.mutables.mutable_value import MutableValue
from pplib.nas.mutators.dynamic_mutator import DynamicMutator


class TestDynamicLinear(TestCase):

    def setUp(self) -> None:
        warnings.simplefilter('ignore', ResourceWarning)

        in_features = MutableValue([4, 6, 8])
        out_features = MutableValue([15, 17, 19])
        self.model = DynamicLinear(in_features, out_features, bias=True)
        self.mutator = DynamicMutator()
        self.mutator.prepare_from_supernet(self.model)

    def test_dynamic_linear(self):
        # test forward during searching
        self.mutator.sample_value(mode='max')
        ins = torch.randn(4, 8)
        outs = self.model(ins)
        print(outs.shape)

        self.mutator.sample_value(mode='min')
        with pytest.raises(RuntimeError):
            outs = self.model(ins)
        print(self.mutator.search_group)

        # test forward fix
        self.mutator.fix_chosen(self.model)
        print(self.model.weight.shape)
        ins = torch.randn(4, 4)
        outs = self.model.forward_fixed(ins)
        print(outs.shape)


if __name__ == '__main__':
    unittest.main()
