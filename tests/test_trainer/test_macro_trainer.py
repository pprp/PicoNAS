import unittest
from unittest import TestCase

from pplib.models import MacroBenchmarkSuperNet
from pplib.trainer import MacroTrainer


class TestMacroTrainer(TestCase):

    def test_get_subnet_flops(self):

        model = MacroBenchmarkSuperNet()

        trainer = MacroTrainer(
            model=model,
            mutator=None,
        )

        fair_list = trainer._generate_fair_list()

        for fair_subnet in fair_list:
            subnet_flops = trainer.get_subnet_flops(fair_subnet)
            print(f'geno: {fair_subnet} -> flops: {subnet_flops}')


if __name__ == '__main__':
    unittest.main()
