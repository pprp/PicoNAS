import unittest
from unittest import TestCase

import torch

from pplib.models import DiffNB301Network, OneShotNB301Network
from pplib.nas.mutators import DiffMutator, OneShotMutator


class TestNB301(TestCase):

    def test_forward_os_nb301(self):

        m = OneShotNB301Network()
        v = OneShotMutator(with_alias=True)
        v.prepare_from_supernet(m)
        print(v.search_group)

        i = torch.randn(4, 3, 32, 32)
        o = m(i)
        print(o.shape)

    def test_forward_diff_nb301(self):

        m = DiffNB301Network()
        v = DiffMutator(with_alias=True)
        v.prepare_from_supernet(m)
        print(v.search_group)

        i = torch.randn(4, 3, 32, 32)
        o = m(i)
        print(o.shape)


if __name__ == '__main__':
    unittest.main()
