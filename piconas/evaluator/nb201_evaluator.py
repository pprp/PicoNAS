from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from nas_201_api import NASBench201API as API
from tqdm import tqdm

from piconas.datasets.predictor.nb201_dataset import Nb201DatasetPINAT
from piconas.evaluator.base import Evaluator
from piconas.nas.mutators import DiffMutator, OneShotMutator
from piconas.predictor.pinat.model_factory import create_best_nb201_model
from piconas.predictor.pruners.predictive import find_measures
from piconas.utils.rank_consistency import (concordant_pair_ratio, kendalltau,
                                            minmax_n_at_k, p_at_tb_k, pearson,
                                            rank_difference, spearman)


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
                 num_sample: int = 50,
                 dataset: str = 'cifar10',
                 type: str = 'eval_acc1es',
                 is_predictor=False,
                 **kwargs):
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
        elif dataset == 'ImageNet16-120':
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
        self.api = API(
            '/data/lujunl/pprp/bench/NAS-Bench-201-v1_1-096897.pth',
            verbose=False)

        if is_predictor is not None:
            # build dataloader for predictor

            # build predictor model
            self.predictor = create_best_nb201_model()
            ckpt_dir = 'checkpoints/nasbench_201/201_cifar10_ParZCBMM_mse_t781_vall_e153_bs10_best_nb201_run2_tau0.783145_ckpt.pt'
            self.predictor.load_state_dict(
                torch.load(ckpt_dir, map_location=torch.device('cpu')))
            self.predictor_dataset = Nb201DatasetPINAT(
                split='all', data_type='test', data_set='cifar10')

    def get_predictor_score(self, genotype):
        """get predictor score for a subnet."""
        # subnet_dict = self.generate_subnet(genotype)
        ss_index = self.query_index(genotype)
        # ss_index = self.generate_genotype(subnet_dict, self.trainer.mutator)
        input = self.predictor_dataset.get_batch(ss_index)

        key_list = [
            'num_vertices', 'lapla', 'edge_num', 'features', 'zcp_layerwise'
        ]
        input['edge_index_list'] = [input['edge_index_list']]
        input['operations'] = torch.tensor(
            input['operations']).unsqueeze(0).unsqueeze(0)

        for _key in key_list:
            if isinstance(input[_key], (list, float, int)):
                input[_key] = torch.tensor(input[_key])
                input[_key] = torch.unsqueeze(input[_key], dim=0)
            elif isinstance(input[_key], np.ndarray):
                input[_key] = torch.from_numpy(input[_key])
                input[_key] = torch.unsqueeze(input[_key], dim=0)
            elif isinstance(input[_key], torch.Tensor):
                input[_key] = torch.unsqueeze(input[_key], dim=0)
            else:
                raise NotImplementedError(
                    f'key: {_key} is not list, is a {type(input[_key])}')
            input[_key] = input[_key]
        score = self.predictor(input)
        return score.item()

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

    def generate_subnet(self, genotype: str):
        """|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|"""
        assert genotype is not None
        genotype = genotype.replace('|+|', '|')
        geno_list = genotype.split('|')[1:-1]
        subnet_dict = dict()
        for i, geno in enumerate(geno_list):
            subnet_dict[i] = geno.split('~')[0]
        return subnet_dict

    def query_result(self, genotype: str, cost_key: str = 'flops'):
        """query the indictor by genotype."""
        dataset = self.trainer.dataset
        index = self.api.query_index_by_arch(genotype)
        # TODO
        xinfo = self.api.get_more_info(index, 'cifar10-valid', hp='200')
        # TODO
        return xinfo['valid-accuracy']

    def query_index(self, genotype: str):
        """query the index by genotype."""
        return self.api.query_index_by_arch(genotype)

    def compute_rank_consistency(self, dataloader,
                                 mutator: OneShotMutator) -> List:
        """compute rank consistency of different types of indicators."""
        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []
        flops_indicator_list: List[float] = []
        # for cpr
        subtract_true_list: List[float] = []
        subtract_indicator_list: List[float] = []

        self.trainer.logger.info('Begin to compute rank consistency...')
        num_sample = 50 if self.num_sample is None else self.num_sample

        for _ in tqdm(range(num_sample)):
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
            score = self.trainer.metric_score(dataloader, random_subnet_dict)
            generated_indicator_list.append(score)

            # get flops
            flops_result = self.query_result(genotype, cost_key='flops')
            flops_indicator_list.append(flops_result)

            # sample another subnet pair for cpr
            random_subnet_dict2 = self.trainer.mutator.random_subnet
            self.trainer.mutator.set_subnet(random_subnet_dict2)
            genotype = self.generate_genotype(random_subnet_dict,
                                              self.trainer.mutator)
            results2 = self.query_result(genotype)  # type is eval_acc1es
            subtract_true_list.append(results - results2)

            score2 = self.trainer.metric_score(dataloader, random_subnet_dict)
            subtract_indicator_list.append(score - score2)

        return self.calc_results(true_indicator_list, generated_indicator_list,
                                 flops_indicator_list, subtract_true_list,
                                 subtract_indicator_list)

    def compute_overall_rank_consistency(self, dataloader) -> List:
        """compute rank consistency of all search space."""
        true_indicator_list: List[float] = []
        generated_indicator_list: List[float] = []
        flops_indicator_list: List[float] = []

        self.trainer.logger.info('Begin to compute rank consistency...')

        for genotype in list(self.api.keys()):
            if 'none' in genotype:
                continue

            subnet_dict = self.generate_subnet(genotype)

            # get true indictor by query nb201 api
            results = self.query_result(genotype)  # type is eval_acc1es
            true_indicator_list.append(results)

            # get score based on supernet
            results = self.trainer.metric_score(dataloader, subnet_dict)
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

        return self.calc_simple_results(true_indicator_list,
                                        generated_indicator_list)

    def calc_results(self,
                     true_indicator_list,
                     generated_indicator_list,
                     flops_indicator_list=None,
                     subtract_true_list=None,
                     subtract_indicator_list=None):

        kt = kendalltau(true_indicator_list, generated_indicator_list)
        ps = pearson(true_indicator_list, generated_indicator_list)
        sp = spearman(true_indicator_list, generated_indicator_list)
        minn_at_ks = minmax_n_at_k(true_indicator_list,
                                   generated_indicator_list)
        patks = p_at_tb_k(true_indicator_list, generated_indicator_list)

        cpr = -1
        if subtract_indicator_list is not None:
            cpr = concordant_pair_ratio(true_indicator_list,
                                        generated_indicator_list)

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
            f"Kendall's tau: {kt}, pearson coeff: {ps}, spearman coeff: {sp}, rank diff: {rd_list}, cpr: {cpr}."
        )

        return kt, ps, sp, rd_list, minn_at_ks, patks, cpr

    def calc_simple_results(self, true_indicator_list,
                            generated_indicator_list):

        kt = kendalltau(true_indicator_list, generated_indicator_list)
        ps = pearson(true_indicator_list, generated_indicator_list)
        sp = spearman(true_indicator_list, generated_indicator_list)
        return kt, ps, sp
