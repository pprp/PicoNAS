from typing import List, Union

import jahs_bench
import numpy as np
import torch.nn.functional as F
from nas_201_api import NASBench201API as API
from tqdm import tqdm

from piconas.evaluator.base import Evaluator
from piconas.nas.mutators import DiffMutator, OneShotMutator
from piconas.predictor.pruners.predictive import find_measures
from piconas.utils.rank_consistency import (concordant_pair_ratio, kendalltau,
                                            minmax_n_at_k, p_at_tb_k, pearson,
                                            rank_difference, spearman)


class JAHSEvaluator(Evaluator):
    """Evaluate the NB201 Benchmark

    Args:
        trainer (NB201Trainer): _description_
        num_sample (int, optional): _description_. Defaults to None.
        search_space (str, optional): _description_. Defaults to 'nasbench201'.
        dataset (str, optional): _description_. Defaults to 'cifar10'.
        type (str, optional): _description_. Defaults to 'eval_acc1es'.
    """

    __known_metrics_in_jahs = ('FLOPS', 'latency', 'runtime', 'size_MB',
                               'test-acc', 'train-acc', 'valid-acc')
    __known_metrics_in_nb201 = ('train_losses', 'eval_losses', 'train_acc1es',
                                'eval_acc1es', 'cost_info')
    __candidate_ops = ('skip_connect', 'none', 'nor_conv_1x1', 'nor_conv_3x3',
                       'avg_pool_3x3')

    def __init__(self,
                 trainer,
                 num_sample: int = None,
                 dataset: str = 'cifar10',
                 nb201_type: str = 'eval_acc1es',
                 jahs_type: str = 'test-acc',
                 **kwargs):
        super().__init__(trainer=trainer, dataset=dataset)
        self.trainer = trainer
        self.num_sample = num_sample
        self.nb201_type = nb201_type 
        self.jahs_type = jahs_type
        self.search_space = 'nasbench201'
        self.dataset = dataset

        if dataset == 'cifar10':
            self.num_classes = 10

        assert jahs_type in self.__known_metrics_in_jahs, \
            f'Not support type {jahs_type}.'
        # assert nb201_type in self.__known_metrics_in_nb201, \
        #     f'Not support type {nb201_type}.'

        # self.nb201_api = API(
        #     './data/benchmark/NAS-Bench-201-v1_0-e61699.pth', verbose=False)
        self.jahs_api = jahs_bench.Benchmark(task='cifar10', save_dir="./data/", download=False)

    def convert_cfg2subnt(self, cfg: dict) -> dict:
        """ Convert the cfg to the subnet dict of mutator."""
        return {
            i - 1: self.__candidate_ops[cfg[f'Op{i}']]
            for i in range(1, 7)
        }

    def convert_subnet2genostr(self, subnet_dict: dict,
                          mutator: Union[OneShotMutator, DiffMutator]) -> str:
        """subnet_dict represent the subnet dict of mutator."""
        # Please make sure that the mutator have been called the
        # `prepare_from_supernet` function.
        alias2group_id = mutator.alias2group_id
        genotype = ''
        for i, (k, v) in enumerate(subnet_dict.items()):
            alias_name = list(alias2group_id.keys())[k]
            rank = alias_name.split('_')[1][-1]  # 0 or 1 or 2
            genotype += '|'
            genotype += f'{v}~{rank}'
            genotype += '|'
            if i in [0, 2]:
                genotype += '+'
        return genotype.replace('||', '|')

    def convert_genostr2subnet(self, genotype: str):
        """|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|"""
        assert genotype is not None
        genotype = genotype.replace('|+|', '|')
        geno_list = genotype.split('|')[1:-1]
        return {i: geno.split('~')[0] for i, geno in enumerate(geno_list)}
    
    def convert_genostr2genoobj(self, genotype: str):
        from examples.jahs_nb201.nasbench201.genos import Structure
        structure = Structure.str2structure(genotype)
        return structure.tolist(remove_str="")[0]


    def query_jahs_result(self, config: dict):
        """query result by config.
        {200: {'runtime': 32470.873046875,
        'size_MB': 0.024951867759227753,
        'valid-acc': 74.22746276855469,
        'latency': 0.3497772514820099,
        'FLOPS': 2.98705792427063,
        'test-acc': 76.1904296875,
        'train-acc': 75.95040893554688}}
        """
        results = self.jahs_api(config, nepochs=200)
        return results[200][self.jahs_type]

    # def query_nb201_result(self, genotype: str, cost_key: str = 'flops'):
    #     """query the indictor by genotype."""
    #     dataset = self.trainer.dataset
    #     index = self.nb201_api.query_index_by_arch(genotype)
    #     # TODO
    #     xinfo = self.nb201_api.get_more_info(index, 'cifar10-valid', hp='200')
    #     # TODO
    #     return xinfo['valid-accuracy']

    def compute_rank_consistency(self, dataloader,
                                 mutator: OneShotMutator) -> List:
        """compute rank consistency of different types of indicators."""
        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []

        self.trainer.logger.info('Begin to compute rank consistency...')
        num_sample = 50 if self.num_sample is None else self.num_sample

        for _ in range(num_sample):          
            cfg = self.jahs_api.sample_config()
            results = self.query_jahs_result(cfg)  # type is eval_acc1es
            true_indicator_list.append(results)

            # get score based on supernet
            subnet_dict = self.convert_cfg2subnt(cfg)
            score = self.trainer.metric_score(dataloader, subnet_dict, cfg)
            generated_indicator_list.append(score)

        return self.calc_results(true_indicator_list, generated_indicator_list)

    def compute_rank_by_predictive(self,
                                   dataloader=None,
                                   measure_name: List = ['flops']) -> List:
        """compute rank consistency by zero cost metric."""
        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []
        flops_indicator_list: List[float] = []
        # for cpr
        subtract_true_list: List[float] = []
        subtract_indicator_list: List[float] = []

        if dataloader is None:
            from piconas.datasets import build_dataloader
            dataloader = build_dataloader('cifar10', 'train')

        num_sample = 50 if self.num_sample is None else self.num_sample

        for _ in tqdm(range(num_sample)):
            # sample random subnet by mutator
            random_subnet_dict = self.trainer.mutator.random_subnet
            self.trainer.mutator.set_subnet(random_subnet_dict)

            # get true indictor by query nb201 api
            genotype = self.generate_genotype(random_subnet_dict,
                                              self.trainer.mutator)
            results = self.query_jahs_result(genotype)  # type is eval_acc1es
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
            tmp_type = self.nb201_type
            self.nb201_type = 'cost_info'
            results = self.query_jahs_result(genotype, cost_key='flops')
            flops_indicator_list.append(results)
            self.nb201_type = tmp_type

            # add another pair for cpr
            random_subnet_dict2 = self.trainer.mutator.random_subnet
            self.trainer.mutator.set_subnet(random_subnet_dict2)
            genotype = self.generate_genotype(random_subnet_dict,
                                              self.trainer.mutator)
            results2 = self.query_jahs_result(genotype)  # type is eval_acc1es
            subtract_true_list.append(results - results2)

            score2 = find_measures(
                self.trainer.model,
                dataloader,
                dataload_info=dataload_info,
                measure_names=measure_name,
                loss_fn=F.cross_entropy,
                device=self.trainer.device)
            subtract_indicator_list.append(score - score2)

        return self.calc_results(true_indicator_list, generated_indicator_list,
                                 flops_indicator_list, subtract_true_list,
                                 subtract_indicator_list)

    def calc_results(self,
                     true_indicator_list,
                     generated_indicator_list):

        kt = kendalltau(true_indicator_list, generated_indicator_list)
        ps = pearson(true_indicator_list, generated_indicator_list)
        sp = spearman(true_indicator_list, generated_indicator_list)
        return [kt, ps, sp]
