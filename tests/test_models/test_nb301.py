import unittest
from unittest import TestCase

import torch

from pplib.models import DiffNASBench301Network, OneShotNASBench301Network
from pplib.nas.mutators import DiffMutator, OneShotMutator


class TestNB301(TestCase):

    def test_forward_os_nb301(self):

        m = OneShotNASBench301Network()
        osm = OneShotMutator(with_alias=True)
        osm.prepare_from_supernet(m)

        a2g = osm.alias2group_id
        g2a = dict(zip(a2g.values(), a2g.keys()))
        for k, value in osm.search_group.items():
            print(k, len(value), g2a[k], value[0])

        random_subnet = osm.random_subnet

        for k, value in random_subnet.items():
            print(k, value, g2a[k])

        i = torch.randn(4, 3, 32, 32)
        o = m(i)
        print(o.shape)

    def test_forward_diff_nb301(self):

        m = DiffNASBench301Network()
        v = DiffMutator(with_alias=True)
        v.prepare_from_supernet(m)
        print(v.search_group)

        i = torch.randn(4, 3, 32, 32)
        o = m(i)
        print(o.shape)

    def test_nb301_api(self):
        from pplib.utils.get_dataset_api import get_dataset_api
        api = get_dataset_api('nasbench301', dataset='cifar10')
        api = api['nb301_model']


if __name__ == '__main__':
    unittest.main()
