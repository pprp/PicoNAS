import unittest
from unittest import TestCase

import torch
import torch.nn as nn

from pplib.nas.mutables import OneShotChoiceRoute, OneShotOP
from pplib.nas.mutators import OneShotMutator


class TestOneShot(TestCase):

    def test_oneshotop(self):

        candidate_ops = nn.ModuleDict()

        candidate_ops.add_module('candidate1', nn.Conv2d(32, 32, 3, 1, 1))
        candidate_ops.add_module('candidate2', nn.Conv2d(32, 32, 5, 1, 2))
        candidate_ops.add_module('candidate3', nn.Conv2d(32, 32, 7, 1, 3))

        osop = OneShotOP(candidate_ops=candidate_ops)

        inputs = torch.randn(4, 32, 32, 32)

        outputs = osop(inputs)
        print(outputs.shape)
        assert outputs is not None

    def test_oneshot_choiceroute(self):
        candidate_ops = nn.ModuleDict()
        # add three edges
        candidate_ops.add_module('candidate1', nn.Conv2d(32, 32, 3, 1, 1))
        candidate_ops.add_module('candidate2', nn.Conv2d(32, 32, 5, 1, 2))
        candidate_ops.add_module('candidate3', nn.Conv2d(32, 32, 7, 1, 3))

        oscr = OneShotChoiceRoute(edges=candidate_ops)
        osm = OneShotMutator()
        osm.prepare_from_supernet(oscr)

        print(osm.search_group)

        inputs = [torch.randn(4, 32, 32, 32) for _ in range(3)]
        o = oscr(inputs)
        print(o.shape)


if __name__ == '__main__':
    unittest.main()
