import unittest
from unittest import TestCase

import pytest
import torch

from pplib.nas.mutables.dynamic import DynamicLinear
from pplib.nas.mutables.dynamic.dynamic_linear import LinearSample


class TestDynamicLinear(TestCase):

    def test_linear(self):
        head = DynamicLinear(48, 10)

        # test sample parameter
        choice = LinearSample(24, 5)
        head.sample_parameters(choice)

        self.assertEqual(head._choice.sample_in_dim, 24)
        self.assertEqual(head._choice.sample_out_dim, 5)

        # test forward
        dummy_input = torch.randn(4, 24)
        head.forward(dummy_input)
        head.forward_choice(dummy_input, choice)

        # test forward all
        dummy_input = torch.randn(4, 48)
        head.forward_all(dummy_input)

        # test out of range assert
        choice = LinearSample(50, 11)
        with pytest.raises(AssertionError):
            head.sample_parameters(choice)

        # test fixed mode
        chosen = LinearSample(32, 10)
        head.fix_chosen(chosen)

        self.assertTrue(head.is_fixed)
        dummy_input = torch.randn(4, 32)
        head.forward_fixed(dummy_input)


if __name__ == '__main__':
    unittest.main()
