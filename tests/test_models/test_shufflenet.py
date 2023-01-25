import unittest
from unittest import TestCase

import torch

from nanonas.models import SearchableShuffleNetV2
from nanonas.models.spos.spos_modules import ShuffleModule, ShuffleXModule


class TestShuffleNet(TestCase):

    def test_shufflenetv2(self):
        model = SearchableShuffleNetV2()
        inputs = torch.randn(4, 3, 32, 32)
        assert model(inputs) is not None

    def test_shuffleModule(self):
        model1 = ShuffleXModule(32, 64, 1)
        model2 = ShuffleModule(32, 64, kernel=3, stride=1)
        model3 = ShuffleModule(32, 64, kernel=5, stride=1)
        model4 = ShuffleModule(32, 64, kernel=7, stride=1)

        inputs = torch.randn(4, 64, 32, 32)
        assert model2(inputs) is not None
        assert model3(inputs) is not None
        assert model4(inputs) is not None
        assert model1(inputs) is not None


if __name__ == '__main__':
    unittest.main()
