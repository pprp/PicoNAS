import unittest
from unittest import TestCase

from pplib.utils.get_dataset_api import get_dataset_api


class TestNasBench(TestCase):

    def test_nas_bench_201(self):

        api = get_dataset_api(search_space='nasbench201', dataset='cifar10')

        tk = list(api['nb201_data'].keys())[1]

        dct = api['nb201_data'][tk]['cifar10-valid']

        for k, v in dct.items():
            print(k, v[-5:] if isinstance(v, list) else v)


if __name__ == '__main__':
    unittest.main()
