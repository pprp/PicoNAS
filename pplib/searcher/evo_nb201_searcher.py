import random
import sys
from typing import Dict, List

import numpy as np
import torch

from pplib.trainer.nb201_trainer import NB201Trainer
from pplib.utils.logging import get_logger

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

sys.setrecursionlimit(10000)


class EvolutionSearcher(object):
    """_summary_

    Args:
        max_epochs (int, optional): _description_. Defaults to 20.
        select_num (int, optional): _description_. Defaults to 10.
        population_num (int, optional): _description_. Defaults to 50.
        mutate_prob (float, optional): _description_. Defaults to 0.1.
        crossover_num (int, optional): _description_. Defaults to 25.
        mutation_num (int, optional): _description_. Defaults to 25.
        flops_limit (float, optional): _description_. Defaults to 300.
        trainer (NB201Trainer, optional): _description_. Defaults to None.
        train_loader (_type_, optional): _description_. Defaults to None.
        val_loader (_type_, optional): _description_. Defaults to None.
    """

    def __init__(
            self,
            max_epochs: int = 20,
            select_num: int = 10,
            population_num: int = 50,
            mutate_prob: float = 0.1,
            crossover_num: int = 25,
            mutation_num: int = 25,
            flops_limit: float = 330 * 1e6,
            trainer: NB201Trainer = None,
            model_path: str = None,  # noqa: E501
            train_loader=None,
            val_loader=None):

        self.max_epochs = max_epochs
        self.select_num = select_num
        self.population_num = population_num
        self.mutate_prob = mutate_prob
        self.crossover_num = crossover_num
        self.mutation_num = mutation_num
        self.flops_limit = flops_limit
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = self.trainer.model
        self.logger = get_logger(name='evoluation_searcher')

        self.logger.info('Loading model weights....')
        if model_path is None:
            model_path = './checkpoints/test_nb201_spos/test_nb201_spos_macro_ckpt_0001.pth.tar'

        state_dict = torch.load(model_path)['state_dict']
        self.model.load_state_dict(state_dict)

        self.memory = []
        # visited
        self.vis_dict = {}
        # select 10 (`select_num`) archs from 50 archs.
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        # candidate subnet configs
        self.candidates = []

    def is_legal(self, subnet: Dict):
        """Judge whether the subnet is visited."""
        if str(subnet) not in self.vis_dict:
            self.vis_dict[str(subnet)] = {}
        info = self.vis_dict[str(subnet)]

        if 'visited' in info:
            return False

        if 'flops' not in info:
            info['flops'] = self.trainer.get_subnet_flops(subnet)

        if info['flops'] > self.flops_limit:
            self.logger.info('flops limit exceed')
            return False
        info['err'] = self.trainer.get_subnet_error(subnet, self.train_loader,
                                                    self.val_loader)
        info['visited'] = True
        return True

    def update_top_k(self, candidates: List, k: int, key, reverse=False):
        """k is 50 or `select_num`."""
        assert k in self.keep_top_k
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def yield_random_subnet(self):
        while True:
            current_subnet = self.trainer.mutator.random_subnet
            # Dict is not a hashable type
            if str(current_subnet) not in self.vis_dict:
                self.vis_dict[str(current_subnet)] = {}
                yield current_subnet

    def get_random(self, num):
        """Get `num` random subnets."""
        self.logger.info('random select ........')
        subnet_iter = self.yield_random_subnet()
        while len(self.candidates) < num:
            subnet = next(subnet_iter)
            if not self.is_legal(subnet):
                continue
            self.candidates.append(subnet)

    def get_mutation(self, k, mutation_num, mutate_prob):
        assert k in self.keep_top_k
        self.logger.info('mutation ......')
        res = []

        max_iters = mutation_num * 10
        subnet_iter = self.yield_random_subnet()

        def mutate_subnet(subnet: Dict):
            convert_to = [
                'conv_3x3', 'skip_connect', 'conv_1x1', 'avg_pool_3x3'
            ]
            for key, value in subnet.items():
                if np.random.random_sample() < mutate_prob:
                    subnet[key] = random.sample(convert_to, 1)[0]
            return subnet

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            subnet = next(subnet_iter)
            subnet = mutate_subnet(subnet)
            if not self.is_legal(subnet):
                continue
            res.append(subnet)

        self.logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        self.logger.info('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def crossover_subnet(subnet1: Dict, subnet2: Dict):
            new_subnet = {}
            for (k1, v1), (k2, v2) in zip(subnet1.items(), subnet2.items()):
                assert k1 == k2
                new_subnet[k1] = random.choice([v1, v2])
            return new_subnet

        subnet_iter = self.yield_random_subnet()
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            subnet1 = next(subnet_iter)
            subnet2 = next(subnet_iter)
            subnet = crossover_subnet(subnet1, subnet2)

            if not self.is_legal(subnet):
                continue
            res.append(subnet)

        self.logger.info('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        self.logger.info(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'
            .format(
                self.population_num, self.select_num, self.mutation_num,
                self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num,
                self.max_epochs))

        # Append `population_num` subnet to candidates.
        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            self.logger.info(f'epoch = {self.epoch}')

            self.memory.append([])
            for subnet in self.candidates:
                self.memory[-1].append(subnet)

            self.update_top_k(
                self.candidates,
                k=self.select_num,
                key=lambda x: self.vis_dict[str(x)]['err'])
            self.update_top_k(
                self.candidates,
                k=50,
                key=lambda x: self.vis_dict[str(x)]['err'])

            self.logger.info(
                f'epoch = {self.epoch} : top {len(self.keep_top_k[50])} result'
            )
            for i, subnet in enumerate(self.keep_top_k[50]):
                self.logger.info(
                    f'No.{i+1} {subnet} Top-1 err = {self.vis_dict[str(subnet)]["err"]}'
                )

            mutation = self.get_mutation(self.select_num, self.mutation_num,
                                         self.mutate_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.candidates.sort(
                key=lambda x: self.vis_dict[str(x)]['err'], reverse=False)
            best_subnet = self.candidates[0]
            genotype = self.trainer.evaluator.generate_genotype(best_subnet, self.trainer.mutator)
            results = self.trainer.evaluator.query_result(genotype, cost_key='eval_acc1es')
            self.logger.info(f"Best Subnet: {best_subnet} Genotype: {genotype} Eval Acc: {results}")
            
