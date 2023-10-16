import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from piconas.predictor.nas_embedding_suite.nb101_ss import NASBench101

import unittest
from unittest.mock import patch, Mock
import numpy as np
import torch

class TestNASBench101(unittest.TestCase):

    def setUp(self):
        self.nasbench101 = NASBench101(path='dummy_path')

    @patch('torch.load', return_value={})
    @patch('json.load', return_value={})
    @patch('os.path.exists', return_value=False)
    def test_init(self, mock_exists, mock_json_load, mock_torch_load):
        nasbench = NASBench101(path='/data2/dongpeijie/share/bench/predictor_embeddings/embedding_datasets/nasbench_only108.tfrecord')
        self.assertIsInstance(nasbench, NASBench101)

    @patch('NB1API.NASBench.get_metrics_from_hash', return_value={
        'module_adjacency': [[1, 0], [0, 1]],
        'module_operations': ['input', 'output']
    })
    def test_get_adj_op(self, mock_get_metrics):
        expected_output = {
            'module_adjacency': [[1, 0], [0, 1]],
            'module_operations': [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
        }
        output = self.nasbench101.get_adj_op(0)
        self.assertEqual(output, expected_output)

    # Mocking the necessary methods for the get_zcp method
    @patch('NB1API.NASBench.get_metrics_from_hash', return_value={
        'module_adjacency': [[1, 0], [0, 1]],
        'module_operations': ['input', 'output']
    })
    def test_get_zcp(self, mock_get_metrics):
        self.nasbench101.zcp_nb101 = {
            'cifar10': {
                'tuple1': {
                    'epe_nas': 0.5,
                    'fisher': 0.2,
                    # ... other values ...
                }
            }
        }
        self.nasbench101.hash_iterator_list = ['hash1', 'hash2']
        expected_output = [0.5, 0.2]  # and other values...
        output = self.nasbench101.get_zcp(0)
        self.assertEqual(output, expected_output)

    # Add more tests ...

if __name__ == '__main__':
    unittest.main()
