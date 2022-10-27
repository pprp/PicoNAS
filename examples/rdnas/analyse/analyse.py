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

# ckpt_path = '/home/stack/project/spos-cifar/examples/rank_nb201/checkpoints/normal_nb201_fairnas_fairsampling_exp-rerun/normal_nb201_fairnas_fairsampling_exp-rerun_nb201_ckpt_0191.pth.tar'

# ckpt_path = '/home/stack/project/spos-cifar/examples/rank_nb201/checkpoints/pairwise-ranking_nb201_spos_adaptive-sampling_exp-rerun/pairwise-ranking_nb201_spos_adaptive-sampling_exp-rerun_nb201_ckpt_0191.pth.tar'

ckpt_path = '/home/stack/project/spos-cifar/examples/rank_nb201/checkpoints/normal_nb201_spos_uniform-sampling_exp-rerun/normal_nb201_spos_uniform-sampling_exp-rerun_nb201_ckpt_0191.pth.tar'

model.load_state_dict(torch.load(ckpt_path)['state_dict'])
mutator = OneShotMutator(with_alias=True)
mutator.prepare_from_supernet(model)

dataloader = build_dataloader(
    'cifar10', 'train', data_dir='../../../data/cifar')
val_dataloader = build_dataloader(
    'cifar10', 'val', data_dir='../../../data/cifar')
trainer = NB201Trainer(model=model, mutator=None, device=torch.device('cpu'))
evaluator = NB201Evaluator(trainer, 50)

zen_model = OneShotNASBench201Network(with_residual=False)
zen_mutator = OneShotMutator(with_alias=True)
zen_mutator.prepare_from_supernet(model)


def flops_dist(dct1, dct2):
    flops1 = trainer.get_subnet_flops(dct1)
    flops2 = trainer.get_subnet_flops(dct2)
    return flops1 - flops2


def calc_gt_list(dct1, dct2):
    results1 = evaluator.query_result(
        evaluator.generate_genotype(dct1, trainer.mutator))
    results2 = evaluator.query_result(
        evaluator.generate_genotype(dct2, trainer.mutator))
    return results1 - results2


def calc_os_list(dct1, dct2):
    results1 = trainer.get_subnet_error(dct1, dataloader, val_dataloader)
    results2 = trainer.get_subnet_error(dct2, dataloader, val_dataloader)
    return results1 - results2


def calc_zerocost_dist(dct1, dct2, zc_proxy='zen'):
    # print("Current zerocost proxy is:", zc_proxy)
    dataload_info = ['random', 1, 10]
    device = torch.device('cpu')

    current_model = None
    current_mutator = None
    if zc_proxy == 'zen':
        current_model = zen_model
        current_mutator = zen_mutator
    else:
        current_model = model
        current_mutator = mutator

    current_mutator.set_subnet(dct1)
    zc1 = find_measures(
        net_orig=current_model,
        dataloader=dataloader,
        dataload_info=dataload_info,
        measure_names=[zc_proxy],
        loss_fn=F.cross_entropy,
        device=device)

    current_mutator.set_subnet(dct2)
    zc2 = find_measures(
        net_orig=current_model,
        dataloader=dataloader,
        dataload_info=dataload_info,
        measure_names=[zc_proxy],
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


def calculate_concordant(list1, list2, name1: str, name2: str):
    total_number = len(list1)
    num_concordant = 0
    for i in range(len(list1)):
        if list1[i] * list2[i] > 0:
            num_concordant += 1
    res = num_concordant / (total_number + 1e-9)
    print(f'Concordant of {name1} and {name2} is: {res}')


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
    calculate_rk(gt_list, zc_list, 'groudtruth', 'zerocost')

    plot_rk(gt_list, hm_dst_list, 'groudtruth', 'hamming')
    plot_rk(gt_list, ad_dst_list, 'groudtruth', 'adaptive')
    plot_rk(gt_list, flops_list, 'groudtruth', 'flops')
    plot_rk(gt_list, zc_list, 'groudtruth', 'zerocost')


def measure_one_shot_concordant(num_samples=200):
    gt_list = []
    os_list = []

    for _ in tqdm(range(num_samples)):
        sg1 = mutator.random_subnet
        sg2 = mutator.random_subnet
        gt_list.append(calc_gt_list(sg1, sg2))
        os_list.append(calc_os_list(sg1, sg2))

    calculate_concordant(gt_list, os_list, 'groudtruth', 'one-shot-fairnas')


def measure_concordant(dist_type: str = None, threshold=None, num_samples=200):
    print(
        f'Current distance type is: {dist_type}, current threshold is: {threshold}'
    )
    hm_dst_list = []
    ad_dst_list = []
    gt_list = []
    zc_dict = {
        'params': [],
        'fisher': [],
        'nwot': [],
        'synflow': [],
        'snip': [],
        'zen': [],
    }
    flops_list = []

    for _ in tqdm(range(num_samples)):
        sg1 = mutator.random_subnet
        sg2 = mutator.random_subnet

        if dist_type == 'hamming':
            if hamming_dist(sg1, sg2) < threshold:
                continue
        elif dist_type == 'adaptive':
            if adaptive_dist(sg1, sg2) < threshold:
                continue

        flops_list.append(flops_dist(sg1, sg2))
        ad_dst_list.append(adaptive_dist(sg1, sg2))
        hm_dst_list.append(adaptive_dist(sg1, sg2))
        gt_list.append(calc_gt_list(sg1, sg2))
        for k in list(zc_dict.keys()):
            zc_dict[k].append(calc_zerocost_dist(sg1, sg2, k))

    calculate_concordant(gt_list, hm_dst_list, 'groudtruth', 'hamming')
    calculate_concordant(gt_list, ad_dst_list, 'groudtruth', 'adaptive')
    calculate_concordant(gt_list, flops_list, 'groudtruth', 'flops')
    for k in list(zc_dict.keys()):
        calculate_concordant(gt_list, zc_dict[k], 'groudtruth', k)


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
    calculate_rk(gt_list, zc_list, 'groudtruth', 'zerocost-adaptive-distance')
    plot_rk(gt_list, flops_list, 'groudtruth', 'flops-adaptive-distance')
    plot_rk(gt_list, zc_list, 'groudtruth', 'zerocost-adaptive-distance')


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
    calculate_rk(gt_list, zc_list, 'groudtruth', 'zerocost-hamming_dist')
    plot_rk(gt_list, flops_list, 'groudtruth', 'flops-hamming_dist')
    plot_rk(gt_list, zc_list, 'groudtruth', 'zerocost-hamming_dist')


if __name__ == '__main__':
    # for t in [1, 3, 5, 7, 9, 11]:
    #     measure_concordant(dist_type='adaptive', threshold=t)

    # for t in [1, 3, 5, 7, 9, 11]:
    #     measure_concordant(dist_type='hamming', threshold=t)

    measure_one_shot_concordant(num_samples=50)
