import unittest
from unittest import TestCase

from pplib.models import OneShotNASBench301Network
from pplib.nas.mutators import OneShotMutator


class TestNB301Trainer(TestCase):

    def test_nb301_evaluator(self):

        m = OneShotNASBench301Network()
        v = OneShotMutator(with_alias=True)
        v.prepare_from_supernet(m)

        print(v.search_group)


if __name__ == '__main__':
    unittest.main()
