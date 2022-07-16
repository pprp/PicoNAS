import unittest
from unittest import TestCase

import pytest
import torch

from pplib.nas.mutables.dynamic import DynamicLayerNorm
from pplib.nas.mutables.dynamic.dynamic_layernorm import LayerNormSample


class TestDynamicLN(TestCase):

    def test_dynamic_layernorm(self):
        m = DynamicLayerNorm(100)
        dummy_input = torch.randn(4, 100)

        # test forward all
        assert m.forward_all(dummy_input) is not None

        # test forward choice
        dummy_input = torch.randn(4, 50)
        choice = LayerNormSample(50)
        assert m.forward_choice(dummy_input, choice) is not None

        choice = LayerNormSample(40)
        m.sample_parameters(choice)
        dummy_input = torch.randn(4, 40)
        assert m.forward_choice(dummy_input) is not None

        # test fix chosen
        chosen = LayerNormSample(30)
        m.fix_chosen(chosen)
        dummy_input = torch.randn(4, 30)
        assert m.forward_fixed(dummy_input) is not None

        # test attribute error
        with pytest.raises(AttributeError):
            m.fix_chosen(choice)


if __name__ == '__main__':
    unittest.main()
