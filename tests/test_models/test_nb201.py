import unittest
from unittest import TestCase

import pytest

from pplib.models import DiffNASBench201Network, OneShotNASBench201Network
from pplib.nas.mutators import DiffMutator, OneShotMutator


class TestNasBench201(TestCase):

    # def test_oneshot_nb201(self):
    #     model = OneShotNASBench201Network(16, 5)
    #     osmutator = OneShotMutator(with_alias=True)
    #     osmutator.prepare_from_supernet(model)

    #     print(osmutator.random_subnet)
    #     print(osmutator.search_group)

    def test_diff_nb201(self):
        model = DiffNASBench201Network(16, 5)
        dfmutator = DiffMutator(with_alias=True)
        import ipdb
        ipdb.set_trace()
        dfmutator.prepare_from_supernet(model)

        print(dfmutator.search_group)


if __name__ == '__main__':
    unittest.main()
