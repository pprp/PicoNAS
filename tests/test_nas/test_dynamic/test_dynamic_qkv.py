import unittest
from unittest import TestCase

import torch

from pplib.nas.mutables.dynamic.dynamic_qkv import DynamicQKV, QKVSample


class TestDynamicQKV(TestCase):

    def test_dynamic_qkv(self):
        dummy_input = torch.randn(4, 48)
        m = DynamicQKV(48, 30)

        # test forward all
        out = m.forward_all(dummy_input)
        print(out.shape)

        # test forward choice
        choice = QKVSample(20, 20)
        dummy_input = torch.randn(4, 20)
        out = m.forward_choice(dummy_input, choice)
        print(out.shape)

        m.sample_parameters(choice)
        out = m.forward_choice(dummy_input)
        print(out.shape)

        # test fix choice
        choice = QKVSample(10, 10)
        dummy_input = torch.randn(4, 10)
        m.fix_chosen(choice)
        out = m.forward_fixed(dummy_input)
        print(out.shape)


if __name__ == '__main__':
    unittest.main()
