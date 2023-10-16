import unittest
from unittest import TestCase


class TestNASBench201(TestCase):

    def setUp(self):
        self.nasbench201 = NASBench201(path='dummy_path')

    def test_get_adjmlp_zcp(self):
        out = self.nasbench201.get_zcp(0)
        print('**** zcp:', out)

    def test_get_cate(self):
        out = self.nasbench201.get_cate(0)
        print('**** cate:', out)

    def test_get_arch2vec(self):
        out = self.nasbench201.get_arch2vec(0)
        print('**** arch2vec:', out)

    def test_get_valacc(self):
        # Assuming some expected value for the validation accuracy
        expected_valacc = 0.85
        self.nasbench201.valacc_list = [expected_valacc]

        out = self.nasbench201.get_valacc(0)
        print('**** valacc:', out)

        # Assert the expected outcome
        self.assertEqual(out, expected_valacc)

    def test_get_norm_w_d(self):
        out = self.nasbench201.get_norm_w_d(0)
        print('**** norm_w_d:', out)

        # Assert the expected outcome
        self.assertEqual(out, [0, 0])  # The method currently returns [0, 0]

    def test_get_numitems(self):
        mock_list_length = 10
        self.nasbench201.hash_iterator_list = [None] * mock_list_length
        print('**** hash_iterator_list:', self.nasbench201.hash_iterator_list)

        out = self.nasbench201.get_numitems()
        print('**** numitems:', out)

        # Assert the expected outcome
        self.assertEqual(out, mock_list_length)

    def test_transform_nb201_operations(self):
        ops = ['input', 'conv3x3-bn-relu', 'output']
        out = self.nasbench201.transform_nb201_operations(ops)
        print('**** transformed ops:', out)
