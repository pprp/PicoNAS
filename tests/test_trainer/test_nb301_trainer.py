import unittest
from unittest import TestCase

from pplib.models import OneShotNB301Network
from pplib.nas.mutators import OneShotMutator


class TestNB301Trainer(TestCase):

    def test_nb301_evaluator(self):

        m = OneShotNB301Network()
        v = OneShotMutator(with_alias=True)


if __name__ == '__main__':
    unittest.main()
