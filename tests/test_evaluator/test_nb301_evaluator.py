import unittest
from unittest import TestCase

from nanonas.evaluator import NB301Evaluator
from nanonas.models.nasbench301 import OneShotNASBench301Network
from nanonas.nas.mutators import OneShotMutator
from nanonas.trainer import NB301Trainer


class TestNB301Evaluator(TestCase):

    def test_nb301(self):
        nb301_model = OneShotNASBench301Network()
        nb301_mutator = OneShotMutator(with_alias=True)
        nb301_mutator.prepare_from_supernet(nb301_model)

        trainer = NB301Trainer(model=nb301_model, mutator=nb301_mutator)

        nb301_evaluator = NB301Evaluator(trainer=trainer, num_sample=50)

        random_subnet = nb301_mutator.random_subnet
        genotype = nb301_evaluator.generate_genotype(random_subnet,
                                                     nb301_mutator)
        print(nb301_evaluator.query_result(genotype))


if __name__ == '__main__':
    unittest.main()
