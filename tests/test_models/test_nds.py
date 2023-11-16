import json
import os
import random
import unittest

from piconas.utils.get_dataset_api import NDS


class TestNDS(unittest.TestCase):
    def setUp(self) -> None:
        self.nds_api = NDS('Amoeba')
        self.iter_nds = iter(self.nds_api)

    def test_get_network(self):
        uid, network = self.iter_nds.__next__()
        print(uid)

    def test_get_accuracy(self):
        acc = self.nds_api.get_final_accuracy(0)
        print(acc)


if __name__ == '__main__':
    unittest.main()
