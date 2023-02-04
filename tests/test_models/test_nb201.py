import unittest
from unittest import TestCase

import numpy as np
import seaborn as sns

from piconas.datasets import build_dataloader
from piconas.models import DiffNASBench201Network, OneShotNASBench201Network
from piconas.nas.mutators import DiffMutator, OneShotMutator
from piconas.predictor.pruners.predictive import find_measures
from piconas.utils.get_dataset_api import get_dataset_api


class TestNasBench201(TestCase):

    def test_generate_arch(self):
        model = OneShotNASBench201Network(16, 5)
        osmutator = OneShotMutator(with_alias=True)
        osmutator.prepare_from_supernet(model)

        random_subnet_dict = osmutator.random_subnet
        alias2group_id = osmutator.alias2group_id

        arch_string = ''
        for i, (k, v) in enumerate(random_subnet_dict.items()):
            # v = 'nor_conv_3x3'
            alias_name = list(alias2group_id.keys())[k]
            rank = alias_name.split('_')[1][-1]  # 0 or 1 or 2
            arch_string += '|'
            arch_string += f'{v}~{rank}'
            arch_string += '|'
            if i in [0, 2]:
                arch_string += '+'
        arch_string = arch_string.replace('||', '|')

        api = get_dataset_api(search_space='nasbench201', dataset='cifar10')
        print('=' * 30)
        print(list(api['nb201_data'].keys())[:3])
        print('=' * 30)
        if arch_string in api['nb201_data'].keys():
            results = api['nb201_data'][arch_string]['cifar10-valid']
            print(f"train_losses: {results['train_losses'][-5:]}")
            print(f"eval_losses: {results['eval_losses'][-5:]}")
            print(f"train_acc1es: {results['train_acc1es'][-5:]}")
            print(f"eval_acc1es: {results['eval_acc1es'][-5:]}")
            print(f"cost_info: {results['cost_info']}")

        else:
            print(f'{arch_string} is not available')
        print('=' * 30)

    def test_oneshot_nb201(self):
        model = OneShotNASBench201Network(16, 5)
        osmutator = OneShotMutator(with_alias=True)
        osmutator.prepare_from_supernet(model)

        print('random subnet:', osmutator.random_subnet)
        print('search group:', osmutator.search_group.keys())
        print('search group[0]:', osmutator.search_group[0])
        print('alias name:', osmutator.alias2group_id.keys())

    def test_diff_nb201(self):
        model = DiffNASBench201Network(16, 5)
        dfmutator = DiffMutator(with_alias=True)
        dfmutator.prepare_from_supernet(model)

        for param1, param2 in zip(dfmutator.parameters(),
                                  dfmutator.arch_params.values()):
            print(
                f'param1 shape: {param1.shape} ==> param2 shape: {param2.shape}'
            )

        print(dfmutator.search_group.keys())

    def test_calculate_distance(self):
        model = OneShotNASBench201Network()
        mutator = OneShotMutator(with_alias=True)
        mutator.prepare_from_supernet(model)

        from piconas.evaluator import NB201Evaluator
        from piconas.trainer import NB201Trainer

        trainer = NB201Trainer(model=model, mutator=None)
        evaluator = NB201Evaluator(trainer, 50)

        def flops_dist(dct1, dct2):
            flops1 = trainer.get_subnet_flops(dct1)
            flops2 = trainer.get_subnet_flops(dct2)
            return int(abs(flops1 - flops2))

        def calc_eval_dist(dct1, dct2):
            results1 = evaluator.query_result(
                evaluator.generate_genotype(dct1, trainer.mutator))
            results2 = evaluator.query_result(
                evaluator.generate_genotype(dct2, trainer.mutator))
            return int(abs(results1 - results2))

        # mean 4.5 std 1.06
        def dis3(dct1, dct2):
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                dist += 1 if v1 != v2 else 0
            return dist

        # mean 6.7 std 2.23
        def dis4(dct1, dct2):
            """
            Distance between conv is set to 0.5
            Distance between conv and other is set to 2
            Distance between other and other is set to 0.5
            """
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                if v1 == v2:
                    continue
                if 'conv' in v1 and 'conv' in v2:
                    dist += 0.5
                elif 'conv' in v1 and ('skip' in v2 or 'pool' in v2):
                    dist += 2
                elif 'conv' in v2 and ('skip' in v1 or 'pool' in v1):
                    dist += 2
                elif 'skip' in v1 and 'pool' in v2:
                    dist += 0.5
                elif 'skip' in v2 and 'pool' in v1:
                    dist += 0.5
                else:
                    raise NotImplementedError(f'v1: {v1} v2: {v2}')
            return dist

        dst_list = []
        true_list = []
        for i in range(1000):
            sg1 = mutator.random_subnet
            sg2 = mutator.random_subnet
            dst_list.append(flops_dist(sg1, sg2))
            true_list.append(calc_eval_dist(sg1, sg2))

        print('=' * 20)
        from piconas.utils.rank_consistency import (kendalltau, pearson,
                                                    spearman)
        kt = kendalltau(dst_list, true_list)
        ps = pearson(dst_list, true_list)
        sp = spearman(dst_list, true_list)
        print(kt, ps, sp)
        print('=' * 20)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots()
        ax1 = sns.scatterplot(x=list(range(len(dst_list))), y=dst_list)
        plt.savefig('./test_dis_flops.png')

        print(
            f'mean: {np.mean(dst_list)} std: {np.std(dst_list)} max: {max(dst_list)} min: {min(dst_list)}'
        )

    def test_subnet_dist_zero_score(self):
        # for zenscore
        model = OneShotNASBench201Network(with_residual=False)
        mutator = OneShotMutator(with_alias=True)
        mutator.prepare_from_supernet(model)
        dataloader = build_dataloader('cifar10', 'train')

        from piconas.evaluator import NB201Evaluator
        from piconas.trainer import NB201Trainer

        trainer = NB201Trainer(model=model, mutator=None)
        evaluator = NB201Evaluator(trainer, 50)

        def flops_dist(dct1, dct2):
            flops1 = trainer.get_subnet_flops(dct1)
            flops2 = trainer.get_subnet_flops(dct2)
            return int(abs(flops1 - flops2))

        def calc_eval_dist(dct1, dct2):
            results1 = evaluator.query_result(
                evaluator.generate_genotype(dct1, trainer.mutator))
            results2 = evaluator.query_result(
                evaluator.generate_genotype(dct2, trainer.mutator))
            return int(abs(results1 - results2))

        def calc_zerocost_dist(dct1, dct2):
            import torch
            import torch.nn.functional as F
            dataload_info = ['random', 1, 10]
            device = torch.device('cuda')

            mutator.set_subnet(dct2)
            zc1 = find_measures(
                net_orig=model,
                dataloader=dataloader,
                dataload_info=dataload_info,
                measure_names=['zen'],
                loss_fn=F.cross_entropy,
                device=device)

            mutator.set_subnet(dct2)
            zc2 = find_measures(
                net_orig=model,
                dataloader=dataloader,
                dataload_info=dataload_info,
                measure_names=['zen'],
                loss_fn=F.cross_entropy,
                device=device)
            return abs(zc1 - zc2)

        # mean 4.5 std 1.06

        def dis3(dct1, dct2):
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                dist += 1 if v1 != v2 else 0
            return dist

        # mean 6.7 std 2.23
        def dis4(dct1, dct2):
            """
            Distance between conv is set to 0.5
            Distance between conv and other is set to 2
            Distance between other and other is set to 0.5
            """
            dist = 0
            for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
                assert k1 == k2
                if v1 == v2:
                    continue
                if 'conv' in v1 and 'conv' in v2:
                    dist += 0.5
                elif 'conv' in v1 and ('skip' in v2 or 'pool' in v2):
                    dist += 2
                elif 'conv' in v2 and ('skip' in v1 or 'pool' in v1):
                    dist += 2
                elif 'skip' in v1 and 'pool' in v2:
                    dist += 0.5
                elif 'skip' in v2 and 'pool' in v1:
                    dist += 0.5
                else:
                    raise NotImplementedError(f'v1: {v1} v2: {v2}')
            return dist

        dst_list = []
        true_list = []
        zc_list = []

        from tqdm import tqdm
        for i in tqdm(range(100)):
            sg1 = mutator.random_subnet
            sg2 = mutator.random_subnet
            dst_list.append(dis4(sg1, sg2))
            true_list.append(calc_eval_dist(sg1, sg2))
            zc_list.append(calc_zerocost_dist(sg1, sg2))

        print('=' * 20)
        from piconas.utils.rank_consistency import (kendalltau, pearson,
                                                    spearman)
        kt = kendalltau(dst_list, zc_list)
        ps = pearson(dst_list, zc_list)
        sp = spearman(dst_list, zc_list)
        print(kt, ps, sp)
        print('=' * 20)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots()
        # ax1 = sns.scatterplot(x=list(range(len(dst_list))), y=dst_list)
        # ax2 = sns.scatterplot(x=list(range(len(dst_list))), y=zc_list)
        ax = sns.scatterplot(x=dst_list, y=true_list)
        ax.set_title('dis4 vs gt')
        plt.savefig('./test_dis4_vs_gt.png')

        print(
            f'mean: {np.mean(dst_list)} std: {np.std(dst_list)} max: {max(dst_list)} min: {min(dst_list)}'
        )

    def test_delete_op(self):
        model = OneShotNASBench201Network(16, 5)
        osmutator = OneShotMutator(with_alias=True)
        osmutator.prepare_from_supernet(model)

        print(osmutator.search_group[0])
        print('before delete rand subnet: ', osmutator.random_subnet)
        for i, vs in osmutator.search_group.items():
            print(i, len(vs), vs[0].choices)
            # delete
            for v in vs:
                v.shrink_choice('skip_connect')
            print(i, len(vs), vs[0].choices)

        print('after delete rand subnet: ', osmutator.random_subnet)


if __name__ == '__main__':
    unittest.main()
