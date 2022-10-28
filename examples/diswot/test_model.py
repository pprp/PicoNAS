import unittest
from unittest import TestCase

import model  # noqa: F401,F403
from model.mutable import MasterNet
# from model.mutable.utils import pretty_format


class TestMutable(TestCase):

    def test_use_masternet(self):
        plainnet_struct = 'SuperConvK3BNRELU(3,32,1,1)SuperResK1K5K1(32,120,1,40,1)SuperResK1K5K1(120,176,2,32,3)SuperResK1K7K1(176,272,1,24,3)SuperResK1K3K1(272,176,1,56,3)SuperResK1K3K1(176,176,1,64,4)SuperResK1K5K1(176,216,2,40,2)SuperResK1K3K1(216,72,2,56,2)SuperConvK1BNRELU(72,512,1,1)'
        
        # print(pretty_format(plainnet_struct))
        m = MasterNet(plainnet_struct=plainnet_struct)
        print(m)


if __name__ == '__main__':
    unittest.main()
