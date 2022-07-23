import unittest
from unittest import TestCase

import torch

from pplib.models import MacroBenchmarkSuperNet


class TestMacroBench(TestCase):

    def test_macro_benchmark(self):
        model = MacroBenchmarkSuperNet()

        inputs = torch.randn(4, 3, 32, 32)

        print(model)

        print(model(inputs).shape)


if __name__ == '__main__':
    unittest.main()
