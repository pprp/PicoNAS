import unittest
from argparse import ArgumentParser
from unittest import TestCase

from piconas.datasets.predictor.data_factory import create_dataloader


class TestPredictorDataset(TestCase):
    """Tests for `predictor_dataset.py`."""

    def setUp(self):
        parser = ArgumentParser()
        # exp and dataset
        parser.add_argument('--exp_name', type=str, default='PINAT')
        parser.add_argument('--bench', type=str, default='201')
        parser.add_argument('--train_split', type=str, default='100')
        parser.add_argument('--eval_split', type=str, default='all')
        parser.add_argument('--dataset', type=str, default='cifar10')
        # training settings
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--gpu_id', type=int, default=0)
        parser.add_argument('--epochs', default=300, type=int)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--wd', default=1e-3, type=float)
        parser.add_argument('--train_batch_size', default=10, type=int)
        parser.add_argument('--eval_batch_size', default=10240, type=int)
        parser.add_argument('--train_print_freq', default=1e6, type=int)
        parser.add_argument('--eval_print_freq', default=10, type=int)
        args = parser.parse_args()
        train_loader, test_loader, train_set, test_set = create_dataloader(
            args)

        self.train_loader = train_loader

    def test_predictor_dataset(self):
        """Test something."""
        for step, batch in enumerate(self.train_loader):
            print('====================')
            print(batch['zcp_lw'])
            print(batch['zcp_lw'].shape)


if __name__ == '__main__':
    unittest.main()
