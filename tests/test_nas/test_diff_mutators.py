import unittest
from unittest import TestCase

import torch

from piconas.models import DiffNASBench201Network
from piconas.nas.mutators import DiffMutator


class TestDiffMutator(TestCase):

    def setUp(self) -> None:
        self.inputs = torch.randn(4, 3, 32, 32)
        self.model = DiffNASBench201Network()
        self.mutator = DiffMutator(with_alias=True)
        self.mutator.prepare_from_supernet(self.model)

    def test_forward(self):
        out = self.model(self.inputs)
        print(out.shape)


if __name__ == '__main__':
    unittest.main()
