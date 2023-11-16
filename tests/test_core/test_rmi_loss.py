import unittest
from unittest import TestCase

import torch

from piconas.core.losses import RMI_loss


class TestRMILoss(TestCase):
    def test_rmi_loss(self):
        fmap1 = torch.randn(3, 3, 32, 32)
        fmap2 = torch.randn(3, 3, 32, 32)
        Loss = RMI_loss(fmap1.size()[0])
        l = Loss(fmap1, fmap2)
        print(l)

    @unittest.skip('do not test')
    def test_other(self):
        print('test other')


if __name__ == '__main__':
    unittest.main()
