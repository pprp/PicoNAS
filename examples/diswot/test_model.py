import unittest
from unittest import TestCase

import model  # noqa: F401,F403
import torch
import torch.nn.functional as F
from model.mutable import MasterNet
from model.mutable.basic_blocks import _remove_bn_layer_

from pplib.datasets import build_dataloader
# from model.mutable.utils import pretty_format
from pplib.predictor.pruners import predictive


class TestMutable(TestCase):

    def test_use_masternet(self):
        plainnet_struct = 'SuperConvK3BNRELU(3,32,1,1)SuperResK1K5K1(32,120,1,40,1)SuperResK1K5K1(120,176,2,32,3)SuperResK1K7K1(176,272,1,24,3)SuperResK1K3K1(272,176,1,56,3)SuperResK1K3K1(176,176,1,64,4)SuperResK1K5K1(176,216,2,40,2)SuperResK1K3K1(216,72,2,56,2)SuperConvK1BNRELU(72,512,1,1)'

        dataload_info = ['random', 3, 100]

        # print(pretty_format(plainnet_struct))
        net = MasterNet(plainnet_struct=plainnet_struct)
        print(net)

        net.block_list = _remove_bn_layer_(net.block_list)

        dataloader = build_dataloader(
            'cifar100', type='train', data_dir='./data/cifar')

        score = predictive.find_measures(
            net,
            dataloader,
            dataload_info=dataload_info,
            measure_names=['nwot'],
            loss_fn=F.cross_entropy,
            device=torch.device('cpu'))

        print(score)


if __name__ == '__main__':
    unittest.main()
