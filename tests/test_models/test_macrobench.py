import random
import unittest
from typing import Dict, List
from unittest import TestCase

import numpy as np
import torch

from pplib.models import MacroBenchmarkSuperNet
from pplib.nas.mutators import OneShotMutator


class TestMacroBench(TestCase):

    def test_macro_benchmark(self):
        model = MacroBenchmarkSuperNet()

        inputs = torch.randn(4, 3, 32, 32)

        mutator = OneShotMutator()
        mutator.prepare_from_supernet(model)

        sg = mutator.search_group

        print(sg.keys())

        print(mutator.random_subnet)

        print(model(inputs).shape)

    def generate_spos_path(self):
        choices = ['I', '1', '2']
        length = 14

        return {i: random.sample(choices, 1)[0] for i in range(length)}

    def test_generate_fair(self):
        choices = ['I', '1', '2']
        length = 14

        all_list = []
        for _ in range(length):
            all_list.append(random.sample(choices, 3))

        all_list = np.array(all_list).T

        result_list: List[Dict] = []

        for i in range(len(choices)):
            tmp_dict = {}
            for idx, choice in enumerate(all_list[i]):
                tmp_dict[idx] = choice
            result_list.append(tmp_dict)

        print(result_list)

    def test_model_complexity(self):
        import copy

        from mmcv.cnn import get_model_complexity_info

        model = MacroBenchmarkSuperNet()

        # WARNING: must before mutator prepare_from_supernet
        copymodel = copy.deepcopy(model)
        copymodel.eval()
        flops, params = get_model_complexity_info(copymodel, (3, 32, 32))
        print(flops, params)

        # for name, module in copymodel.named_modules():
        #     flops = getattr(module, '__flops__', 0)
        #     if flops > 0:
        #         print(name, flops)

        mutator = OneShotMutator()
        # import ipdb; ipdb.set_trace()
        mutator.prepare_from_supernet(copymodel)

        single_dict = self.generate_spos_path()  # {0: 'xx'}
        current_flops = 0

        for k, v in mutator.search_group.items():
            current_choice = single_dict[k]  # '1' or '2' or 'I'

            choice_flops = 0
            for name, module in v[0]._candidate_ops[
                    current_choice].named_modules():
                flops = getattr(module, '__flops__', 0)
                if flops > 0:
                    # print(name, flops)
                    choice_flops += flops

            print(f'k: {k} choice: {current_choice} flops: {choice_flops}')

            current_flops += choice_flops

        print('Current flops: ', current_flops)


if __name__ == '__main__':
    unittest.main()
