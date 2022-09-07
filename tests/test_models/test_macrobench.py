import random
import unittest
from typing import Dict, List
from unittest import TestCase

import numpy as np
import seaborn as sns
import torch

from pplib.models import MacroBenchmarkSuperNet
from pplib.nas.mutators import OneShotMutator


class TestMacroBench(TestCase):

    def test_calculate_distance(self):
        model = MacroBenchmarkSuperNet()

        mutator = OneShotMutator()
        mutator.prepare_from_supernet(model)

        def dis1(dct1, dct2):
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                dist += 1 if v1 != v2 else 0
            return dist

        def dis2(dct1, dct2):
            """
            Distance between I and 1 2 is set to 2
            Distance between 1 and 2 is set to 0.5
            """
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                if v1 == v2:
                    continue
                if set([v1, v2]) == set(['1', '2']):
                    dist += 0.5
                elif set([v1, v2]) == set(['1', 'I']):
                    dist += 2
                elif set([v1, v2]) == set(['1', '2']):
                    dist += 2
            return dist

        dst_list = []
        for i in range(1000):
            sg1 = mutator.random_subnet
            sg2 = mutator.random_subnet
            dst = dis2(sg1, sg2)
            dst_list.append(dst)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots()
        ax1 = sns.scatterplot(x=list(range(len(dst_list))), y=dst_list)
        plt.savefig('./test_dis2.png')

        print(
            f'mean: {np.mean(dst_list)} std: {np.std(dst_list)} max: {max(dst_list)} min: {min(dst_list)}'
        )

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

    # def test_model_complexity(self):
    #     import copy

    #     from mmcv.cnn import get_model_complexity_info

    #     model = MacroBenchmarkSuperNet()

    #     # WARNING: must before mutator prepare_from_supernet
    #     copymodel = copy.deepcopy(model)
    #     copymodel.eval()
    #     flops, params = get_model_complexity_info(copymodel, (3, 32, 32))
    #     print(flops, params)

    #     # for name, module in copymodel.named_modules():
    #     #     flops = getattr(module, '__flops__', 0)
    #     #     if flops > 0:
    #     #         print(name, flops)

    #     mutator = OneShotMutator()
    #     # import ipdb; ipdb.set_trace()
    #     mutator.prepare_from_supernet(copymodel)

    #     single_dict = self.generate_spos_path()  # {0: 'xx'}
    #     current_flops = 0

    #     for k, v in mutator.search_group.items():
    #         current_choice = single_dict[k]  # '1' or '2' or 'I'

    #         choice_flops = 0
    #         for name, module in v[0]._candidate_ops[
    #                 current_choice].named_modules():
    #             flops = getattr(module, '__flops__', 0)
    #             if flops > 0:
    #                 # print(name, flops)
    #                 choice_flops += flops

    #         print(f'k: {k} choice: {current_choice} flops: {choice_flops}')

    #         current_flops += choice_flops

    #     print('Current flops: ', current_flops)


if __name__ == '__main__':
    unittest.main()
