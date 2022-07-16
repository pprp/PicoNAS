import unittest
from unittest import TestCase

from pplib.nas.mutables.dynamic.dynamic_relativeposition import \
    DynamicRelativePosion2D


class TestDynamicRelativePosition(TestCase):

    def test_dynamic_relative_position(self):
        embed_dim = 256
        num_heads = 8
        max_relative_position = 14
        m = DynamicRelativePosion2D(embed_dim // num_heads,
                                    max_relative_position)

        assert m is not None


if __name__ == '__main__':
    unittest.main()
