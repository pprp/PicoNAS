import unittest
from unittest import TestCase

from pplib.nas.mutables.dynamic.dynamic_relativeposition import \
    DynamicRelativePosion2D


class TestDynamicRelativePosition(TestCase):

    def test_dynamic_relative_position(self):

        m = DynamicRelativePosion2D()

        assert m is not None


if __name__ == '__main__':
    unittest.main()
