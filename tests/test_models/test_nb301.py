import unittest
from unittest import TestCase

import torch

from pplib.models import OneShotNB301Network
from pplib.nas.mutators import OneShotMutator


class TestNB301(TestCase):

    def test_forward_nb301(self):

        m = OneShotNB301Network()
        v = OneShotMutator(with_alias=True)
        v.prepare_from_supernet(m)
        print(v.search_group)

        i = torch.randn(4, 3, 32, 32)
        o = m(i)
        print(o.shape)


if __name__ == '__main__':
    unittest.main()
