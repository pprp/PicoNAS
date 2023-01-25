import unittest
from collections import namedtuple
from unittest import TestCase

import torch

from nanonas.models import DiffNASBench301Network, OneShotNASBench301Network
from nanonas.nas.mutables import OneShotChoiceRoute
from nanonas.nas.mutators import DiffMutator, OneShotMutator

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class TestNB301(TestCase):

    def test_osnb301_choiceroute(self):
        nb301 = OneShotNASBench301Network()
        osm = OneShotMutator(with_alias=True)
        osm.prepare_from_supernet(nb301)

        print(osm.alias2group_id.keys())
        print(len(osm.alias2group_id.keys()))
        print(len(osm.search_group))

        print('=' * 20)
        for name, module in nb301.named_modules():
            if isinstance(module, OneShotChoiceRoute):
                choice_of_route = module.random_choice
                print(name, choice_of_route,
                      module._edges[choice_of_route[0]].random_choice)
                print(module._edges.keys(),
                      list(module._edges.keys()).index(choice_of_route[0]))
        print('=' * 20)

        # build genotype
        for idx, groups in osm.search_group.items():
            print(idx, groups[0])

        def get_group_id_by_module(mutator, module):
            for gid, module_list in mutator.search_group.items():
                if module in module_list:
                    return gid
            return None

        normal_list = []
        reduce_list = []
        currnet_random_subnet = osm.random_subnet
        for idx, choices in currnet_random_subnet.items():
            # print(idx, choice)
            if isinstance(choices, list):
                # choiceroute object
                for choice in choices:
                    if 'normal' in choice:
                        # choice route object
                        c_route = osm.search_group[idx][0]
                        # get current key by index
                        idx_of_op = int(choice[-1])
                        # current_key = normal_n3_p1
                        current_key = list(c_route._edges.keys())[idx_of_op]
                        # get oneshot op
                        os_op = c_route._edges[current_key]
                        # get group id
                        gid = get_group_id_by_module(osm, os_op)
                        choice_str = currnet_random_subnet[gid]
                        assert isinstance(choice_str, str)
                        normal_list.append((choice_str, idx_of_op))
                    elif 'reduce' in choice:
                        # choice route object
                        c_route = osm.search_group[idx][0]
                        # get current key by index
                        idx_of_op = int(choice[-1])
                        current_key = list(c_route._edges.keys())[idx_of_op]
                        # get oneshot op
                        os_op = c_route._edges[current_key]
                        # get group id
                        gid = get_group_id_by_module(osm, os_op)
                        choice_str = currnet_random_subnet[gid]
                        assert isinstance(choice_str, str)
                        reduce_list.append((choice_str, idx_of_op))
        genotype = Genotype(
            normal=normal_list,
            normal_concat=[2, 3, 4, 5],
            reduce=reduce_list,
            reduce_concat=[2, 3, 4, 5])
        print(genotype)

        from nanonas.utils.get_dataset_api import get_dataset_api
        api = get_dataset_api('nasbench301', dataset='cifar10')
        api = api['nb301_model']
        p_m, r_m = api
        p_r = p_m.predict(
            config=genotype, representation='genotype', with_noise=False)
        print('performance:', p_r)

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
        from nanonas.utils.get_dataset_api import get_dataset_api
        api = get_dataset_api('nasbench301', dataset='cifar10')
        api = api['nb301_model']


if __name__ == '__main__':
    unittest.main()
