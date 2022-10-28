import torch
import torch.nn.functional as F
from fixed_models import resnet20, resnet110
from mmcv.cnn import get_model_complexity_info
from mutable_student import mutable_resnet20

from pplib.datasets import build_dataloader
from pplib.predictor.pruners import predictive

# from pplib.utils.rank_consistency import kendalltau, pearson, spearman


def generate_config(mutable_depth=None):
    if mutable_depth is None:
        mutable_depth = [2, 6, 10, 14]
    config_list = []
    len_of_mutable = len(mutable_depth)

    for i in range(len_of_mutable):
        for j in range(len_of_mutable):
            for k in range(len_of_mutable):
                results = [
                    mutable_depth[i], mutable_depth[j], mutable_depth[k]
                ]
                config_list.append(results)
    return config_list


if __name__ == '__main__':
    config_list = generate_config()
    dataloader = build_dataloader(
        'cifar100', type='train', data_dir='./data/cifar')
    dataload_info = ['random', 3, 100]
    # 'epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
    # 'nwot', 'plain', 'snip', 'synflow', 'flops', 'params'

    list_sc = []
    list_dp = []

    for cfg in config_list:
        net = mutable_resnet20(depth_list=cfg, num_classes=100)
        score = predictive.find_measures(
            net,
            dataloader,
            dataload_info=dataload_info,
            measure_names=['nwot'],
            loss_fn=F.cross_entropy,
            device=torch.device('cuda:0'))
        flops, params = get_model_complexity_info(
            net, input_shape=(3, 32, 32), print_per_layer_stat=False)
        print(
            f'subnet cfg: {cfg} score: {score:.2f} \t flops: {flops} \t params: {params}'
        )
        list_dp.append(sum(cfg))
        list_sc.append(score)

    # print(f'kd: {kendalltau(list_dp, list_sc)} pr: {pearson(list_dp, list_sc)} sp: {spearman(list_dp, list_sc)}')

    # teacher resnet110 student resnet20

    tnet = resnet110()
    snet = resnet20()

    score = predictive.find_measures(
        tnet,
        dataloader,
        dataload_info=dataload_info,
        measure_names=['nwot'],
        loss_fn=F.cross_entropy,
        device=torch.device('cuda:0'))
    flops, params = get_model_complexity_info(
        tnet, input_shape=(3, 32, 32), print_per_layer_stat=False)
    print(
        f'teacher net | score: {score:.2f} \t flops: {flops} \t params: {params}'
    )

    score = predictive.find_measures(
        snet,
        dataloader,
        dataload_info=dataload_info,
        measure_names=['nwot'],
        loss_fn=F.cross_entropy,
        device=torch.device('cuda:0'))
    flops, params = get_model_complexity_info(
        snet, input_shape=(3, 32, 32), print_per_layer_stat=False)
    print(
        f'student net | score: {score:.2f} \t flops: {flops} \t params: {params}'
    )
