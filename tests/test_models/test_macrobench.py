import random
import unittest
from typing import Dict, List
from unittest import TestCase, result

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

    def test_generate_fair(self):
        choices = ['I', '1', '2']
        length = 14

        single_path = {i: random.sample(choices, 1)[0] for i in range(length)}

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

        print(single_path)


if __name__ == '__main__':
    unittest.main()
