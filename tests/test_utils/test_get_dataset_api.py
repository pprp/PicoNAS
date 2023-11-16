import unittest
from unittest import TestCase

from piconas.utils.get_dataset_api import get_dataset_api


class TestNasBench(TestCase):
    def setUp(self) -> None:
        self.api = get_dataset_api(
            search_space='nasbench201', dataset='cifar10')

    def test_nas_bench_201(self):
        tk = list(self.api['nb201_data'].keys())[1]
        dct = self.api['nb201_data'][tk]['cifar10-valid']
        for k, v in dct.items():
            print(k, v[-5:] if isinstance(v, list) else v)

    def test_best_results(self):
        """There is something wrong with the best model"""
        max_result = -1
        max_arch_str = ''
        for arch_str in list(self.api['nb201_data'].keys()):
            eval_res_list = self.api['nb201_data'][arch_str]['cifar10-valid'][
                'eval_acc1es'
            ][-5:]
            if max(eval_res_list) > max_result:
                max_result = max(eval_res_list)
                max_arch_str = arch_str
        print(max_result, max_arch_str)

    def test_nas_bench_201_api(self):
        """offical api for D-X-Y"""
        from nas_201_api import NASBench201API as API

        api = API('./data/benchmark/NAS-Bench-201-v1_0-e61699.pth', verbose=False)
        arch_str = '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|'

        index = api.query_index_by_arch(arch_str)
        # dataset = 'cifar10-valid' 'cifar10'
        xinfo = api.get_more_info(index, 'cifar10-valid', hp='200')
        for k, v in xinfo.items():
            print(k, v)

        print(f"valid accuracy: {xinfo['valid-accuracy']}")
        print(f"test accuracy: {xinfo['test-accuracy']}")


if __name__ == '__main__':
    unittest.main()
