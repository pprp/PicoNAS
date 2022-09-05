import argparse
import functools
import os
import random
import sys
import time
from typing import List

import numpy as np
import torch

from pplib.trainer.nb201_trainer import NB201Trainer
from pplib.utils.logging import get_logger

# from flops import get_subnet_flops
# from tester import get_subnet_err # BN Calibration + test

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

sys.setrecursionlimit(10000)


class EvolutionSearcher(object):

    def __init__(self,
                 max_epochs: int = 20,
                 select_num: int = 10,
                 population_num: int = 50,
                 mutate_prob: float = 0.1,
                 crossover_num: int = 25,
                 mutation_num: int = 25,
                 flops_limit: float = 300,
                 trainer: NB201Trainer = None,
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

        # self.model = torch.nn.DataParallel(self.model).cuda()
        # supernet_state_dict = torch.load(
        #     '../Supernet/models/checkpoint-latest.pth.tar')['state_dict']
        # self.model.load_state_dict(supernet_state_dict)
        # self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        # visited
        self.vis_dict = {}
        # select 10 (`select_num`) archs from 50 archs.
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        # candidate subnet configs
        self.candidates = []

    # def save_checkpoint(self):
    #     if not os.path.exists(self.log_dir):
    #         os.makedirs(self.log_dir)
    #     info = {}
    #     info['memory'] = self.memory
    #     info['candidates'] = self.candidates
    #     info['vis_dict'] = self.vis_dict
    #     info['keep_top_k'] = self.keep_top_k
    #     info['epoch'] = self.epoch
    #     torch.save(info, self.checkpoint_name)
    #     print('save checkpoint to', self.checkpoint_name)

    # def load_checkpoint(self):
    #     if not os.path.exists(self.checkpoint_name):
    #         return False
    #     info = torch.load(self.checkpoint_name)
    #     self.memory = info['memory']
    #     self.candidates = info['candidates']
    #     self.vis_dict = info['vis_dict']
    #     self.keep_top_k = info['keep_top_k']
    #     self.epoch = info['epoch']

    #     print('load checkpoint from', self.checkpoint_name)
    #     return True

    def is_legal(self, subnet):
        """Judge whether the subnet is visited."""
        assert isinstance(subnet, tuple) and len(subnet) == self.nr_layer

        if subnet not in self.vis_dict:
            self.vis_dict[subnet] = {}
        info = self.vis_dict[subnet]

        if 'visited' in info:
            return False

        if 'flops' not in info:
            info['flops'] = self.trainer.get_subnet_flops(subnet)

        self.logger.info(subnet, info['flops'])

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
            if current_subnet not in self.vis_dict:
                self.vis_dict[current_subnet] = {}
                yield current_subnet

    def get_random(self, num):
        self.logger.info('random select ........')
        subnet_iter = self.yield_random_subnet()

        while len(self.candidates) < num:
            subnet = next(subnet_iter)
            if not self.is_legal(subnet):
                continue
            self.candidates.append(subnet)
            self.logger.info('random {}/{}'.format(len(self.candidates), num))
        self.logger.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, mutate_prob):
        assert k in self.keep_top_k
        self.logger.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            subnet = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer):
                if np.random.random_sample() < mutate_prob:
                    subnet[i] = np.random.randint(self.nr_state)
            return tuple(subnet)

        subnet_iter = self.stack_random_subnet(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            subnet = next(subnet_iter)
            if not self.is_legal(subnet):
                continue
            res.append(subnet)
            self.logger.info('mutation {}/{}'.format(len(res), mutation_num))

        self.logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        self.logger.info('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))

        subnet_iter = self.stack_random_subnet(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            subnet = next(subnet_iter)
            if not self.is_legal(subnet):
                continue
            res.append(subnet)
            self.logger.info('crossover {}/{}'.format(len(res), crossover_num))

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

        self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            self.logger.info('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for subnet in self.candidates:
                self.memory[-1].append(subnet)

            self.update_top_k(
                self.candidates,
                k=self.select_num,
                key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

            self.logger.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, subnet in enumerate(self.keep_top_k[50]):
                self.logger.info('No.{} {} Top-1 err = {}'.format(
                    i + 1, subnet, self.vis_dict[subnet]['err']))
                ops = [i for i in subnet]
                self.logger.info(ops)

            mutation = self.get_mutation(self.select_num, self.mutation_num,
                                         self.mutate_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        self.save_checkpoint()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--mutate_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
    parser.add_argument('--max-train-iters', type=int, default=200)
    parser.add_argument('--max-test-iters', type=int, default=40)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=200)
    args = parser.parse_args()

    t = time.time()

    searcher = EvolutionSearcher(args)

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))


if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
