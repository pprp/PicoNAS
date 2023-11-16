import unittest
from unittest import TestCase

import torch
import torch.nn as nn

from piconas.models.spos import SearchableMobileNet
from piconas.nas.mutators import OneShotMutator
from piconas.utils.angle.weight_vector import get_mb_arch_vector


class TestAngle(TestCase):
    def test_mb_angle(self):
        model = SearchableMobileNet()
        mutator = OneShotMutator()
        mutator.prepare_from_supernet(model)

        random_dict = mutator.random_subnet

        arch_vector1 = get_mb_arch_vector(
            supernet=model, mutator=mutator, subnet_dict=random_dict
        )

        arch_vector2 = get_mb_arch_vector(
            supernet=model, mutator=mutator, subnet_dict=random_dict
        )

        assert arch_vector1 is not None
        assert arch_vector2 is not None

        cosine = nn.CosineSimilarity(dim=0)

        angle = torch.acos(cosine(arch_vector1, arch_vector2))

        self.assertAlmostEqual(angle, 0)


if __name__ == '__main__':
    unittest.main()
