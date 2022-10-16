import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pplib.datasets import build_dataloader
from pplib.evaluator import NB201Evaluator
from pplib.models import OneShotNASBench201Network
from pplib.nas.mutators import OneShotMutator
from pplib.predictor.pruners.predictive import find_measures
from pplib.trainer import NB201Trainer
from pplib.utils.rank_consistency import kendalltau, pearson, spearman

model = OneShotNASBench201Network()
mutator = OneShotMutator(with_alias=True)
mutator.prepare_from_supernet(model)
dataloader = build_dataloader('cifar10', 'train')
trainer = NB201Trainer(model=model, mutator=None)
evaluator = NB201Evaluator(trainer, 50)


def flops_dist(dct1, dct2):
    flops1 = trainer.get_subnet_flops(dct1)
    flops2 = trainer.get_subnet_flops(dct2)
    return flops1 - flops2


def calc_gt_list(dct1, dct2):
    results1 = evaluator.query_result(
        evaluator.generate_genotype(dct1, trainer.mutator))
    results2 = evaluator.query_result(
        evaluator.generate_genotype(dct2, trainer.mutator))
    return int(abs(results1 - results2))


def calc_zerocost_dist(dct1, dct2):
    dataload_info = ['random', 1, 10]
    device = torch.device('cuda')

    mutator.set_subnet(dct1)
    zc1 = find_measures(
        net_orig=model,
        dataloader=dataloader,
        dataload_info=dataload_info,
        measure_names=['nwot'],
        loss_fn=F.cross_entropy,
        device=device)

    mutator.set_subnet(dct2)
    zc2 = find_measures(
        net_orig=model,
        dataloader=dataloader,
        dataload_info=dataload_info,
        measure_names=['nwot'],
        loss_fn=F.cross_entropy,
        device=device)
    return zc1 - zc2


# mean 4.5 std 1.06
def hamming_dist(dct1, dct2):
    dist = 0
    for (k1, v1), (k2, v2) in zip(dct1.items(), dct2.items()):
        assert k1 == k2
        dist += 1 if v1 != v2 else 0
    return dist


# mean 6.7 std 2.23
def adaptive_dist(dct1, dct2):
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


def calculate_rk(list1, list2, name1: str, name2: str):
    kt = kendalltau(list1, list2)
    ps = pearson(list1, list2)
    sp = spearman(list1, list2)

    res = f'RK of {name1} and {name2} is (kendall tau: {kt} pearson: {ps} spearman: {sp})'
    print(res)
    return res


def plot_rk(list1, list2, name1: str, name2: str):
    ax = sns.scatterplot(x=list1, y=list2)
    ax.set_title(f'{name1} vs {name2}')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    plt.savefig(f'./correlation_{name1}_vs_{name2}.png')
    plt.clf()
    plt.cla()


def main():
    hm_dst_list = []
    ad_dst_list = []
    gt_list = []
    zc_list = []
    flops_list = []

    for _ in tqdm(range(100)):
        sg1 = mutator.random_subnet
        sg2 = mutator.random_subnet

        flops_list.append(flops_dist(sg1, sg2))
        ad_dst_list.append(adaptive_dist(sg1, sg2))
        hm_dst_list.append(adaptive_dist(sg1, sg2))
        gt_list.append(calc_gt_list(sg1, sg2))
        zc_list.append(calc_zerocost_dist(sg1, sg2))

    calculate_rk(gt_list, hm_dst_list, 'groudtruth', 'hamming')
    calculate_rk(gt_list, ad_dst_list, 'groudtruth', 'adaptive')
    calculate_rk(gt_list, flops_list, 'groudtruth', 'flops')
    calculate_rk(gt_list, zc_list, 'groudtruth', 'zenscore')

    plot_rk(gt_list, hm_dst_list, 'groudtruth', 'hamming')
    plot_rk(gt_list, ad_dst_list, 'groudtruth', 'adaptive')
    plot_rk(gt_list, flops_list, 'groudtruth', 'flops')
    plot_rk(gt_list, zc_list, 'groudtruth', 'zenscore')


def sample_with_adaptive_distance():
    gt_list = []
    zc_list = []
    flops_list = []

    for _ in tqdm(range(100)):
        sg1 = mutator.random_subnet
        sg2 = mutator.random_subnet

        if adaptive_dist(sg1, sg2) < 6.7:
            continue

        flops_list.append(flops_dist(sg1, sg2))
        gt_list.append(calc_gt_list(sg1, sg2))
        zc_list.append(calc_zerocost_dist(sg1, sg2))

    calculate_rk(gt_list, flops_list, 'groudtruth', 'flops-adaptive-distance')
    calculate_rk(gt_list, zc_list, 'groudtruth', 'zenscore-adaptive-distance')
    plot_rk(gt_list, flops_list, 'groudtruth', 'flops-adaptive-distance')
    plot_rk(gt_list, zc_list, 'groudtruth', 'zenscore-adaptive-distance')


def sample_with_hamming_distance():
    gt_list = []
    zc_list = []
    flops_list = []

    for _ in tqdm(range(100)):
        sg1 = mutator.random_subnet
        sg2 = mutator.random_subnet

        if hamming_dist(sg1, sg2) < 4.5:
            continue

        flops_list.append(flops_dist(sg1, sg2))
        gt_list.append(calc_gt_list(sg1, sg2))
        zc_list.append(calc_zerocost_dist(sg1, sg2))

    calculate_rk(gt_list, flops_list, 'groudtruth', 'flops-hamming_dist')
    calculate_rk(gt_list, zc_list, 'groudtruth', 'zenscore-hamming_dist')
    plot_rk(gt_list, flops_list, 'groudtruth', 'flops-hamming_dist')
    plot_rk(gt_list, zc_list, 'groudtruth', 'zenscore-hamming_dist')


if __name__ == '__main__':
    main()
    sample_with_hamming_distance()
    sample_with_adaptive_distance()
