from collections import namedtuple
from typing import List, Union

import numpy as np
import torch.nn.functional as F

from pplib.evaluator.base import Evaluator
from pplib.nas.mutators import DiffMutator, OneShotMutator
from pplib.predictor.pruners.predictive import find_measures
from pplib.utils.get_dataset_api import get_dataset_api
from pplib.utils.rank_consistency import (kendalltau, minmax_n_at_k, p_at_tb_k,
                                          pearson, rank_difference, spearman)

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class NB301Evaluator(Evaluator):
    """Evaluate the NB301 Benchmark

    Args:
        trainer (NB301Trainer): _description_
        num_sample (int, optional): _description_. Defaults to None.
        search_space (str, optional): _description_. Defaults to 'nasbench301'.
        dataset (str, optional): _description_. Defaults to 'cifar10'.
        type (str, optional): _description_. Defaults to 'eval_acc1es'.
    """

    def __init__(self,
                 trainer,
                 num_sample: int = None,
                 dataset: str = 'cifar10',
                 type: str = 'eval_acc1es'):
        super().__init__(trainer=trainer, dataset=dataset)
        self.trainer = trainer
        self.num_sample = num_sample
        self.type = type
        self.search_space = 'nasbench301'
        self.dataset = dataset

        if dataset == 'cifar10':
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.num_classes = 100
        elif dataset == 'imagenet16':
            self.num_classes = 120
            self.dataset = 'ImageNet16-120'

        assert type in ['train_losses', 'eval_losses',
                        'train_acc1es', 'eval_acc1es', 'cost_info'], \
            f'Not support type {type}.'
        """
        The key of api is the genotype of nb301
            such as '|avg_pool_3x3~0|+
                     |nor_conv_1x1~0|skip_connect~1|+
                     |nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        The value of api is also a dict:
                  key: cifar10-valid
                  value: train_losses, eval_losses, train_acc1es
                         eval_acc1es, cost_info
        """
        self.api = self.load_benchmark()
        self.performance_model = self.api[0]
        self.runtime_model = self.api[1]

    def load_benchmark(self):
        """load benchmark to get api controller."""
        api = get_dataset_api(self.search_space, self.dataset)
        return api['nb301_model']

    def generate_genotype(self, subnet_dict: dict,
                          mutator: Union[OneShotMutator, DiffMutator]) -> str:
        """subnet_dict represent the subnet dict of mutator."""

        # Please make sure that the mutator have been called the
        # `prepare_from_supernet` function.

        def get_group_id_by_module(mutator, module):
            for gid, module_list in mutator.search_group.items():
                if module in module_list:
                    return gid
            return None

        normal_list = []
        reduce_list = []
        for idx, choices in subnet_dict.items():
            # print(idx, choice)
            if isinstance(choices, list):
                # choiceroute object
                for choice in choices:
                    if 'normal' in choice:
                        # choice route object
                        c_route = mutator.search_group[idx][0]
                        # get current key by index
                        idx_of_op = int(choice[-1])
                        # current_key = normal_n3_p1
                        current_key = list(c_route._edges.keys())[idx_of_op]
                        # get oneshot op
                        os_op = c_route._edges[current_key]
                        # get group id
                        gid = get_group_id_by_module(mutator, os_op)
                        choice_str = subnet_dict[gid]
                        assert isinstance(choice_str, str)
                        normal_list.append((choice_str, idx_of_op))
                    elif 'reduce' in choice:
                        # choice route object
                        c_route = mutator.search_group[idx][0]
                        # get current key by index
                        idx_of_op = int(choice[-1])
                        current_key = list(c_route._edges.keys())[idx_of_op]
                        # get oneshot op
                        os_op = c_route._edges[current_key]
                        # get group id
                        gid = get_group_id_by_module(mutator, os_op)
                        choice_str = subnet_dict[gid]
                        assert isinstance(choice_str, str)
                        reduce_list.append((choice_str, idx_of_op))

        genotype = Genotype(
            normal=normal_list,
            normal_concat=[2, 3, 4, 5],
            reduce=reduce_list,
            reduce_concat=[2, 3, 4, 5])
        return genotype

    def query_result(self, genotype: Genotype):
        """query the indictor by genotype."""
        return self.performance_model.predict(
            config=genotype, representation='genotype', with_noise=False)

    def compute_rank_consistency(self, dataloader,
                                 mutator: OneShotMutator) -> List:
        """compute rank consistency of different types of indicators."""
        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []
        flops_indicator_list: List[float] = []

        self.trainer.logger.info('Begin to compute rank consistency...')
        num_sample = 50 if self.num_sample is None else self.num_sample

        for _ in range(num_sample):
            # sample random subnet by mutator
            random_subnet_dict = mutator.random_subnet

            # get true indictor by query nb301 api
            genotype = self.generate_genotype(random_subnet_dict, mutator)
            results = self.query_result(genotype)  # type is eval_acc1es
            true_indicator_list.append(results)

            # get score based on supernet
            results = self.trainer.metric_score(dataloader, random_subnet_dict)
            generated_indicator_list.append(results)

            # get flops
            flops_result = self.trainer.get_subnet_flops(random_subnet_dict)
            flops_indicator_list.append(flops_result)

        return self.calc_results(true_indicator_list, generated_indicator_list,
                                 flops_indicator_list)

    def compute_rank_by_flops(self) -> List:
        """compute rank consistency based on flops."""
        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []

        self.trainer.logger.info('Begin to compute rank consistency...')
        num_sample = 50 if self.num_sample is None else self.num_sample

        for _ in range(num_sample):
            # sample random subnet by mutator
            random_subnet_dict = self.trainer.mutator.random_subnet

            # get true indictor by query nb301 api
            genotype = self.generate_genotype(random_subnet_dict,
                                              self.trainer.mutator)
            results = self.query_result(genotype)  # type is eval_acc1es
            true_indicator_list.append(results)

            # get score based on flops
            flops_result = self.trainer.get_subnet_flops(random_subnet_dict)
            generated_indicator_list.append(flops_result)

        return self.calc_results(true_indicator_list, generated_indicator_list,
                                 generated_indicator_list)

    def compute_rank_by_predictive(self,
                                   dataloader=None,
                                   measure_name=None) -> List:
        """compute rank consistency by zerocost metric."""
        if dataloader is None:
            from pplib.datasets import build_dataloader
            dataloader = build_dataloader('cifar10', 'train')
        if measure_name is None:
            measure_name = ['flops']

        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []
        flops_indicator_list: List[float] = []

        num_sample = 50 if self.num_sample is None else self.num_sample

        for _ in range(num_sample):
            # sample random subnet by mutator
            random_subnet_dict = self.trainer.mutator.random_subnet

            # get true indictor by query nb301 api
            genotype = self.generate_genotype(random_subnet_dict,
                                              self.trainer.mutator)
            results = self.query_result(genotype)  # type is eval_acc1es
            true_indicator_list.append(results)

            assert isinstance(measure_name, list) and len(measure_name) == 1, \
                f'The measure name should be a list but got {type(measure_name)}' \
                f' and the lenght should be 1 but got {len(measure_name)}'

            dataload_info = ['random', 3, self.num_classes]

            # get score based on zenscore
            self.trainer.mutator.set_subnet(random_subnet_dict)
            score = find_measures(
                self.trainer.model,
                dataloader,
                dataload_info=dataload_info,
                measure_names=measure_name,
                loss_fn=F.cross_entropy,
                device=self.trainer.device)

            generated_indicator_list.append(score)

            # get flops
            flops_result = self.trainer.get_subnet_flops(random_subnet_dict)
            flops_indicator_list.append(flops_result)

        return self.calc_results(true_indicator_list, generated_indicator_list,
                                 flops_indicator_list)

    def calc_results(self,
                     true_indicator_list,
                     generated_indicator_list,
                     flops_indicator_list=None):

        kt = kendalltau(true_indicator_list, generated_indicator_list)
        ps = pearson(true_indicator_list, generated_indicator_list)
        sp = spearman(true_indicator_list, generated_indicator_list)
        minn_at_ks = minmax_n_at_k(true_indicator_list,
                                   generated_indicator_list)
        patks = p_at_tb_k(true_indicator_list, generated_indicator_list)

        if flops_indicator_list is not None:
            # calculate rank difference by range.

            # compute rank diff in 5 section with different flops range.
            sorted_idx_by_flops = np.array(flops_indicator_list).argsort()

            # compute splited index by flops
            range_length = len(sorted_idx_by_flops) // 5
            splited_idx_by_flops = [
                sorted_idx_by_flops[i * range_length:(i + 1) * range_length]
                for i in range(
                    int(len(sorted_idx_by_flops) / range_length) + 1)
                if (sorted_idx_by_flops[i * range_length:(i + 1) *
                                        range_length]).any()
            ]

            true_indicator_list = np.array(true_indicator_list)
            generated_indicator_list = np.array(generated_indicator_list)

            rd_list = []
            for splited_idx_by_flop in splited_idx_by_flops:
                current_idx = np.array(splited_idx_by_flop)
                tmp_rd = rank_difference(true_indicator_list[current_idx],
                                         generated_indicator_list[current_idx])
                rd_list.append(tmp_rd)

        print(
            f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}, rank diff: {rd_list}."
        )

        return kt, ps, sp, rd_list, minn_at_ks, patks
