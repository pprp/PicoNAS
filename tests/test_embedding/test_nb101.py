import unittest
from unittest.mock import patch

from piconas.predictor.nas_embedding_suite.nb101_ss import NASBench101


class TestNASBench101(unittest.TestCase):

    def setUp(self):
        self.nasbench101 = NASBench101(path='dummy_path')

    def test_get_adjmlp_zcp(self):
        out = self.nasbench101.get_zcp(0)
        print('zcp:', out)

    def test_get_cate(self):
        out = self.nasbench101.get_cate(0)
        print('cate:', out)

    def test_get_arch2vec(self):
        out = self.nasbench101.get_arch2vec(0)
        print('arch2vec:', out)

    def test_get_valacc(self):
        # Assuming some expected value for the validation accuracy
        expected_valacc = 0.85
        self.nasbench101.valacc_list = [expected_valacc]

        out = self.nasbench101.get_valacc(0)
        print('valacc:', out)

    def test_get_norm_w_d(self):
        out = self.nasbench101.get_norm_w_d(0)
        print('norm_w_d:', out)

    def test_get_numitems(self):
        mock_list_length = 10
        self.nasbench101.hash_iterator_list = [None] * mock_list_length
        print('hash_iterator_list:', self.nasbench101.hash_iterator_list)

        out = self.nasbench101.get_numitems()
        print('numitems:', out)

    def test_transform_nb101_operations(self):
        ops = ['input', 'conv3x3-bn-relu', 'output']
        out = self.nasbench101.transform_nb101_operations(ops)
        print('transformed ops:', out)

    def test_pad_size_6(self):
        matrix = [[1]]
        ops = ['input', 'output']

        out_matrix, out_ops = self.nasbench101.pad_size_6(matrix, ops)
        print('padded matrix:', out_matrix)
        print('padded ops:', out_ops)

    def test_index_to_embedding(self):
        embed = self.nasbench101.index_to_embedding(0)
        print(embed)


if __name__ == '__main__':
    unittest.main()
