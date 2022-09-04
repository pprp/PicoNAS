import random
import unittest
from unittest import TestCase

from pplib.trainer.nb201_trainer import NB201Trainer
from pplib.trainer.sucessive_halving import (Brick, Level,
                                             SuccessiveHalvingPyramid)


class TestNB201Trainer(TestCase):

    def test_level(self):
        level = Level()

        for _ in range(4):
            b = Brick(
                subnet_cfg={'test': 1},
                num_iters=random.randint(0, 10),
                prior_score=random.randint(0, 10),
                val_acc=random.random())
            level.append(b)

        print('=' * 10)
        print(level)
        print('=' * 10)
        level.sort_by_iters()
        print(level)
        print('=' * 10)
        level.sort_by_prior()
        print(level)
        print('=' * 10)
        level.sort_by_val()
        print(level)
        print('=' * 10)
        print(len(level))
        print(level.pop(0))


if __name__ == '__main__':
    unittest.main()
