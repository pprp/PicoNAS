import json
import random
from typing import Dict, List

from piconas.evaluator.base import Evaluator
from piconas.utils.misc import convert_arch2dict
from piconas.utils.rank_consistency import kendalltau, pearson, spearman


class MacroEvaluator(Evaluator):
    """MacroEvaluator

    Args:
        trainer (_type_): _description_
        bench_path (_type_): _description_
        num_sample (_type_, optional): _description_. Defaults to None.
        type (str, optional): _description_. Defaults to 'test_acc'.
    """

    def __init__(
        self,
        trainer,
        bench_path=None,
        num_sample=None,
        type='test_acc',
        dataset='cifar10',
    ):
        super().__init__(trainer, bench_path)
        assert dataset in {'cifar10', 'cifar100'}
        if bench_path is None:
            if dataset == 'cifar10':
                self.bench_path = './data/benchmark/benchmark_cifar10_dataset.json'
            else:
                self.bench_path = './data/benchmark/benchmark_cifar100_dataset.json'
        else:
            self.bench_path = bench_path

        self.trainer = trainer
        self.num_sample = num_sample
        self.type = type
        assert type in [
            'test_acc',
            'MMACs',
            'val_acc',
            'Params',
        ], f'Not support type {type}.'

        self.bench_dict = self.load_benchmark()

    def load_benchmark(self):
        """load benchmark to get dict."""
        assert self.bench_path.endswith('json')

        with open(self.bench_path, 'r') as f:
            bench_dict = json.load(f)

        if self.num_sample is not None:
            bench_dict = self.sample_archs(bench_dict=bench_dict)

        return bench_dict

    def sample_archs(self, bench_dict) -> Dict:
        sampled_keys = random.sample(bench_dict.keys(), k=self.num_sample)
        return {arch: bench_dict[arch] for arch in sampled_keys}

    def compute_rank_consistency(self, dataloader):
        """compute rank consistency of different types of indicators."""
        true_indicator_list: List[float] = []
        supernet_indicator_list: List[float] = []

        self.trainer.logger.info('Begin to compute rank consistency...')

        for i, (k, v) in enumerate(self.bench_dict.items()):
            self.trainer.logger.info(f'evaluating the {i}th architecture.')

            subnet_dict = convert_arch2dict(k)
            indicator = self.trainer.metric_score(
                dataloader, subnet_dict=subnet_dict)

            supernet_indicator_list.append(indicator)
            true_indicator_list.append(v[self.type])

        kt = kendalltau(true_indicator_list, supernet_indicator_list)
        ps = pearson(true_indicator_list, supernet_indicator_list)
        sp = spearman(true_indicator_list, supernet_indicator_list)

        print(
            f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}.")

        return kt, ps, sp

    def compute_rank_by_flops(self):
        """compute rank consistency of flops"""
        true_indicator_list: List[float] = []
        supernet_indicator_list: List[float] = []

        self.trainer.logger.info('Begin to compute rank consistency...')

        for i, (k, v) in enumerate(self.bench_dict.items()):
            self.trainer.logger.info(f'evaluating the {i}th architecture.')

            subnet_dict = convert_arch2dict(k)
            # indicator = self.trainer.metric_score(
            #     self.dataloader, subnet_dict=subnet_dict)
            indicator = self.trainer.get_subnet_flops(subnet_dict)
            supernet_indicator_list.append(indicator)
            true_indicator_list.append(v[self.type])

        kt = kendalltau(true_indicator_list, supernet_indicator_list)
        ps = pearson(true_indicator_list, supernet_indicator_list)
        sp = spearman(true_indicator_list, supernet_indicator_list)

        print(
            f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}.")
        return kt, ps, sp

    def compute_rank_by_nwot(self):
        """compute rank consistency of nwot"""
        true_indicator_list: List[float] = []
        supernet_indicator_list: List[float] = []

        self.trainer.logger.info('Begin to compute rank consistency...')

        for i, (k, v) in enumerate(self.bench_dict.items()):
            self.trainer.logger.info(f'evaluating the {i}th architecture.')

            subnet_dict = convert_arch2dict(k)
            # indicator = self.trainer.metric_score(
            #     self.dataloader, subnet_dict=subnet_dict)
            indicator = self.trainer.get_subnet_nwot(subnet_dict)
            supernet_indicator_list.append(indicator)
            true_indicator_list.append(v[self.type])

        kt = kendalltau(true_indicator_list, supernet_indicator_list)
        ps = pearson(true_indicator_list, supernet_indicator_list)
        sp = spearman(true_indicator_list, supernet_indicator_list)

        print(
            f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}.")
        return kt, ps, sp
