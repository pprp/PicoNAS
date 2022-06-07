import unittest
from unittest import TestCase

from pplib.models import DiffNASBench201Network, OneShotNASBench201Network
from pplib.nas.mutators import DiffMutator, OneShotMutator


class TestNasBench201(TestCase):

    def test_generate_arch(self):
        model = OneShotNASBench201Network(16, 5)
        osmutator = OneShotMutator(with_alias=True)
        osmutator.prepare_from_supernet(model)

        random_subnet_dict = osmutator.random_subnet
        alias2group_id = osmutator.alias2group_id

        mapping = {
            'conv_3x3': 'nor_conv_3x3',
            'skip_connect': 'skip_connect',
            'conv_1x1': 'nor_conv_1x1',
            'avg_pool_3x3': 'avg_pool_3x3',
            'none': 'none',
        }

        arch_string = ''

        for k, v in random_subnet_dict.items():
            # v = 'conv_3x3'
            mapped_op_name = mapping[v]
            alias_name = list(alias2group_id.keys())[k]
            rank = alias_name.split('_')[1][-1]  # 0 or 1 or 2
            arch_string += '|'
            arch_string += mapped_op_name + '~' + rank
            arch_string += '|'

        print(arch_string)

    def test_oneshot_nb201(self):
        model = OneShotNASBench201Network(16, 5)
        osmutator = OneShotMutator(with_alias=True)
        osmutator.prepare_from_supernet(model)

        print('random subnet:', osmutator.random_subnet)
        print('search group:', osmutator.search_group.keys())
        print('alias name:', osmutator.alias2group_id.keys())

    def test_diff_nb201(self):
        model = DiffNASBench201Network(16, 5)
        dfmutator = DiffMutator(with_alias=True)
        dfmutator.prepare_from_supernet(model)

        print(dfmutator.search_group.keys())


if __name__ == '__main__':
    unittest.main()
