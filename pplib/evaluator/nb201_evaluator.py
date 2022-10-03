from typing import List, Union

import numpy as np
import torch.nn.functional as F

from pplib.evaluator.base import Evaluator
from pplib.nas.mutators import DiffMutator, OneShotMutator
from pplib.predictor.pruners.predictive import find_measures
from pplib.utils.get_dataset_api import get_dataset_api
from pplib.utils.rank_consistency import (kendalltau, minmax_n_at_k, p_at_tb_k,
                                          pearson, rank_difference, spearman)


class NB201Evaluator(Evaluator):
    """Evaluate the NB201 Benchmark

    Args:
        trainer (NB201Trainer): _description_
        num_sample (int, optional): _description_. Defaults to None.
        search_space (str, optional): _description_. Defaults to 'nasbench201'.
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
        self.search_space = 'nasbench201'
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
        The key of api is the genotype of nb201
            such as '|avg_pool_3x3~0|+
                     |nor_conv_1x1~0|skip_connect~1|+
                     |nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
        The value of api is also a dict:
                  key: cifar10-valid
                  value: train_losses, eval_losses, train_acc1es
                         eval_acc1es, cost_info
        """
        self.api = self.load_benchmark()

    def load_benchmark(self):
        """load benchmark to get api controller."""
        api = get_dataset_api(self.search_space, self.dataset)
        return api['nb201_data']

    def generate_genotype(self, subnet_dict: dict,
                          mutator: Union[OneShotMutator, DiffMutator]) -> str:
        """subnet_dict represent the subnet dict of mutator."""
        # Please make sure that the mutator have been called the
        # `prepare_from_supernet` function.
        alias2group_id = mutator.alias2group_id
        genotype = ''
        for i, (k, v) in enumerate(subnet_dict.items()):
            # v = 'nor_conv_3x3'
            alias_name = list(alias2group_id.keys())[k]
            rank = alias_name.split('_')[1][-1]  # 0 or 1 or 2
            genotype += '|'
            genotype += f'{v}~{rank}'
            genotype += '|'
            if i in [0, 2]:
                genotype += '+'
        genotype = genotype.replace('||', '|')
        return genotype

    def query_result(self, genotype: str, cost_key: str = 'flops'):
        """query the indictor by genotype."""
        dataset = self.trainer.dataset
        if dataset == 'cifar10':
            dataset = 'cifar10-valid'
        elif dataset == 'cifar100':
            dataset = 'cifar100'
        elif dataset == 'imagenet16':
            dataset = 'ImageNet16-120'
        else:
            raise NotImplementedError(f'Not Support dataset type:{dataset}')

        # TODO default datasets is cifar10, support other dataset in the future.
        if self.type in [
                'train_losses', 'eval_losses', 'train_acc1es', 'eval_acc1es'
        ]:
            return self.api[genotype][dataset][self.type][-1]
        elif self.type == 'cost_info':
            # example:
            # cost_info: {'flops': 78.56193, 'params': 0.559386,
            #             'latency': 0.01493, 'train_time': 10.21598}
            assert cost_key is not None
            return self.api[genotype][dataset][self.type][cost_key]
        else:
            raise f'Not supported type: {self.type}.'

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
            random_subnet_dict_ = mutator.random_subnet

            # process for search space shrink and expand
            random_subnet_dict = dict()
            for k, v in random_subnet_dict_.items():
                random_subnet_dict[k] = v.rstrip('_')

            # get true indictor by query nb201 api
            genotype = self.generate_genotype(random_subnet_dict, mutator)
            results = self.query_result(genotype)  # type is eval_acc1es
            true_indicator_list.append(results)

            # get score based on supernet
            results = self.trainer.metric_score(dataloader, random_subnet_dict)
            generated_indicator_list.append(results)

            # get flops
            flops_result = self.query_result(genotype, cost_key='flops')
            flops_indicator_list.append(flops_result)

        return self.calc_results(true_indicator_list, generated_indicator_list,
                                 flops_indicator_list)

    def compute_rank_by_predictive(self,
                                   dataloader=None,
                                   measure_name: List = ['flops']) -> List:
        """compute rank consistency by zero cost metric."""
        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []
        flops_indicator_list: List[float] = []

        if dataloader is None:
            from pplib.datasets import build_dataloader
            dataloader = build_dataloader('cifar10', 'train')

        self.trainer.logger.info('Begin to compute rank consistency...')
        num_sample = 50 if self.num_sample is None else self.num_sample

        for _ in range(num_sample):
            # sample random subnet by mutator
            random_subnet_dict = self.trainer.mutator.random_subnet

            # get true indictor by query nb201 api
            genotype = self.generate_genotype(random_subnet_dict,
                                              self.trainer.mutator)
            results = self.query_result(genotype)  # type is eval_acc1es
            true_indicator_list.append(results)

            assert isinstance(measure_name, list) and len(measure_name) == 1, \
                f'The measure name should be a list but got {type(measure_name)}' \
                f' and the lenght should be 1 but got {len(measure_name)}'

            dataload_info = ['random', 3, self.num_classes]

            # get predict indicator by predictive.
            score = find_measures(
                self.trainer.model,
                dataloader,
                dataload_info=dataload_info,
                measure_names=measure_name,
                loss_fn=F.cross_entropy,
                device=self.trainer.device)
            generated_indicator_list.append(score)

            # get score based on flops
            tmp_type = self.type
            self.type = 'cost_info'
            results = self.query_result(genotype, cost_key='flops')
            flops_indicator_list.append(results)
            self.type = tmp_type

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
