import unittest
from unittest import TestCase

import torch

from pplib.nas.mutables.dynamic.dynamic_embedding import (DynamicPatchEmbed,
                                                          PatchSample)


class TestDynamicEmbed(TestCase):

    def test_dynamic_embed(self):
        imgs = torch.randn(4, 3, 224, 224)
        m = DynamicPatchEmbed(224, 16, 3, 768)

        # test forward all
        out = m.forward_all(imgs)
        print(out.shape)

        # test forward choice
        choice = PatchSample(256)
        out = m.forward_choice(imgs, choice)
        print(out.shape)

        m.sample_parameters(choice)
        m.forward_choice(imgs)

        # test forward fix
        choice = PatchSample(100)
        m.fix_chosen(choice)
        out = m.forward_fixed(imgs)
        print(out.shape)


if __name__ == '__main__':
    unittest.main()
