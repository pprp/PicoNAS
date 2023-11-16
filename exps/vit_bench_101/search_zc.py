import argparse
import copy
import csv
import gc
import math
import os
import random
import time
from typing import Union

import numpy as np
import pycls.core.config as config
import pycls.core.logging as logging
import pycls.datasets.loader as data_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from autozc.structures import GraphStructure, LinearStructure, TreeStructure
from autozc.utils.rank_consistency import kendalltau, pearson, spearman
from moe import MLP, MOE
from pycls.core.config import cfg
from pycls.datasets.loader import _DATASETS
from pycls.models.build import MODEL
from pycls.predictor.pruners.predictive import find_measures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import Tensor

logger = logging.get_logger(__name__)


def all_same(items):
    return all(x == items[0] for x in items)


def build_model(arch_config, cfg, num_classes):
    model = MODEL.get(cfg.MODEL.TYPE)(
        arch_config=arch_config, num_classes=num_classes)
    return model


def is_anomaly(zc_score: Union[torch.Tensor, float, int, tuple] = None) -> bool:
    """filter the score with -1,0,nan,inf"""
    if isinstance(zc_score, tuple):
        return False
    if isinstance(zc_score, Tensor):
        zc_score = zc_score.item()
    if (
        zc_score is None
        or zc_score == -1
        or math.isnan(zc_score)
        or math.isinf(zc_score)
        or zc_score == 0
    ):
        return True
    return False


def is_anomaly_group(zc_group) -> bool:
    """filter the score with -1,0,nan,inf"""
    for item in zc_group:
        if is_anomaly(item):
            return True
    return False


def joint_func(rank_score):
    # return sum(rank_score)/len(rank_score)
    return sum([abs(item) for item in rank_score]) / len(rank_score)


def pad(layer_zc):
    item_len = [len(m) for m in layer_zc]
    for item in layer_zc:
        if len(item) != max(item_len):
            for i in range(max(item_len) - len(item)):
                item.append(0)
    return layer_zc


def standrd(data):
    min_value = torch.min(data)
    max_value = torch.max(data)
    res = (data - min_value) / (max_value - min_value)
    return res


def obtain_gt(gt_results, gt_num, vanilla=False):
    new_res, arch_pop = [], []
    acc_pop = [[] for _ in range(len(list(_DATASETS.keys())))]

    index_list = random.sample(range(len(gt_results)), gt_num)

    for idx in index_list:
        new_res.append(gt_results[idx])

    for item in new_res:
        arch_pop.append(item['arch'])
        for data_idx, dataset in enumerate(list(_DATASETS.keys())):
            if vanilla:
                acc_pop[data_idx].append(item[dataset]['base'])
            else:
                acc_pop[data_idx].append(item[dataset]['kd'])

    return arch_pop, acc_pop


def moe(num_experts, input_data, target_data, num_epochs, test_data):
    prediction = [[] for _ in range(len(test_data))]
    model = MOE(len(input_data[0]), num_experts, MLP)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(torch.Tensor(input_data))
        # 计算损失
        loss = criterion(outputs, torch.Tensor(target_data))
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    for i in range(len(test_data)):
        prediction[i] = model(torch.Tensor(test_data[i]))
    return prediction


def parzc_fitness(cfg, data_loader, arch_pop, acc_pop, structure, num_classes):
    """structure is belong to popultion."""

    gt_score = acc_pop
    zc_score = []

    layer_zc = []

    data_iter = iter(data_loader)
    try:
        img, label, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        img, label, _ = next(data_iter)

    for arch_idx, arch_cfg in enumerate(arch_pop):
        temp_model = build_model(arch_cfg, cfg, num_classes)
        if torch.cuda.is_available():
            temp_model.cuda()
        zc = structure(img, label, temp_model)

        if zc == -1:
            return -1
        if is_anomaly_group(zc):
            return -1

        # early exit
        if len(zc_score) > 3 and all_same(zc_score):
            return -1

        layer_zc.append(zc)
        zc_final = sum(zc) / len(zc)
        logger.info(
            f'The {arch_idx}-th {cfg.MODEL.TYPE} arch: autoprox score: {zc_final}'
        )
        zc_score.append(zc_final)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # TODO add inf check
    if len(zc_score) <= 1 or np.isnan(kendalltau(gt_score, zc_score)):
        return -1

    # release memory
    del img, label, temp_model
    torch.cuda.empty_cache()
    gc.collect()
    return layer_zc


def data_process(x_train, y_train, data_num, ratio):
    moe_train_x, moe_train_y, moe_test_x, moe_test_y = [], [], [], []
    for i in range(data_num):
        x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(
            pad(x_train[i]), y_train[i], test_size=1 - ratio, random_state=42
        )
        moe_train_x += x_train_temp
        moe_train_y += y_train_temp
        moe_test_x.append(x_test_temp)
        moe_test_y.append(y_test_temp)
    return moe_train_x, moe_train_y, moe_test_x, moe_test_y


def autoprox_parzc(cfg, data_loader, arch_pop, acc_pop, struct, ratio):
    if struct.kendall_score != -1:
        return struct.kendall_score

    rank_score, ken = [], []
    moe_train_x, moe_train_y = [], []
    for i in range(len(acc_pop)):
        # classes =[100, 102, 4, 200, 10]
        classes = [100, 102, 10]
        single_data_layerzc = parzc_fitness(
            cfg, data_loader[i], arch_pop, acc_pop[i], struct, num_classes=classes[i]
        )

        if single_data_layerzc == -1:
            logger.info('Invalid structure, exit...')
            return -1

        single_data_zc = [sum(item) / len(item)
                          for item in single_data_layerzc]
        single_data_kendall = kendalltau(acc_pop[i], single_data_zc)
        logger.info(
            f'kendall score on dataset {list(_DATASETS.keys())[i]} is : {single_data_kendall}'
        )
        rank_score.append(single_data_kendall)

        moe_train_x.append(single_data_layerzc)
        moe_train_y.append(acc_pop[i])

    if struct.kendall_score == -1 and not is_anomaly_group(rank_score):
        logger.info(
            f'Before moe, kendall score for datasets {list(_DATASETS.keys())[:len(acc_pop)]}: {rank_score}'
        )

    moe_train_x, moe_train_y, moe_test_x, moe_test_y = data_process(
        moe_train_x, moe_train_y, len(data_loader), ratio
    )

    predictions = moe(
        len(data_loader), moe_train_x, moe_train_y, num_epochs=100, test_data=moe_test_x
    )

    for i in range(len(data_loader)):
        kendalltau_temp = kendalltau(
            moe_test_y[i], predictions[i].detach().numpy())
        spearman_temp = spearman(
            moe_test_y[i], predictions[i].detach().numpy())
        pearson_temp = pearson(moe_test_y[i], predictions[i].detach().numpy())
        ken.append(kendalltau_temp)

    if struct.kendall_score == -1 and not is_anomaly_group(ken):
        logger.info(
            f'After moe, kendall score for datasets {list(_DATASETS.keys())[:len(acc_pop)]}: {ken}'
        )
        joint_kdall = joint_func(ken)
        logger.info(
            f'Valid structure, joint correlation metric is : {joint_kdall}')
        struct.kendall_score = joint_kdall
        return joint_kdall


def random_search(cfg, args, arch_pop, acc_pop, data_loader, structure):
    population = []
    pop_zc = []
    logger.info('Initialize population')
    while len(population) < args.popu_size:
        struct = structure(args.tree_node)
        logger.info(
            f'Struct={struct} Input={struct.genotype["input_geno"]} Op={struct.genotype["op_geno"]}'
        )
        score = autoprox_parzc(
            cfg, data_loader, arch_pop, acc_pop, struct, args.train_ratio
        )
        if is_anomaly(score):
            continue
        population.append(struct)
        pop_zc.append(score)
        logger.info(f'Current population size: {len(population)}')

    np_pop_zc = np.array(pop_zc)
    argidxs = np.argsort(np_pop_zc)[::-1]

    # best structure on the run
    running_struct = population[argidxs[0]]
    logger.info(
        f'Best Kendall Tau: {pop_zc[argidxs[0]]} Struct={running_struct} Input={running_struct.genotype["input_geno"]} Op={running_struct.genotype["op_geno"]}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evo search zc',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./work_dirs/Search_Autoprox_Parzc',
        help='log path',
    )
    parser.add_argument(
        '--refer_cfg',
        default='./configs/parzc/autoformer/autoformer-ti-subnet_c100_base.yaml',
        type=str,
        help='save output path',
    )
    parser.add_argument(
        '--tree_node', default=3, type=int, help='number of nodes in tree'
    )
    # popu size
    parser.add_argument(
        '--popu_size',
        default=10,
        type=int,
        help='population size should be larger than 10',
    )
    parser.add_argument('--gt_path', type=str, default=None,
                        help='ground truth path')
    parser.add_argument(
        '--gt_num', type=int, default=100, help='number of ground truth'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8, help='ratio of gt used to train gbdt'
    )
    parser.add_argument(
        '--vanilla', action='store_true', help='search under vanilla gt'
    )

    args = parser.parse_args()
    config.load_cfg(args.refer_cfg)
    config.assert_cfg()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    logging.setup_logging()
    time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    log_file = os.path.join(
        args.save_dir,
        '{}_{}_{}_{}.txt'.format(
            cfg.MODEL.TYPE, cfg.AUTO_PROX.type, cfg.PROXY_DATASET, time_str
        ),
    )

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logger.info(message)

    data_loader = data_loader.construct_proxy_loader()

    gt_results = torch.load(args.gt_path)
    arch_pop, acc_pop = obtain_gt(gt_results, args.gt_num, args.vanilla)

    structure = None

    if cfg.AUTO_PROX.type == 'linear':
        structure = LinearStructure
    elif cfg.AUTO_PROX.type == 'tree':
        structure = TreeStructure
    elif cfg.AUTO_PROX.type == 'graph':
        structure = GraphStructure

    logger.info('Random Search AutoProxParZC...')
    t1 = time.time()

    random_search(cfg, args, arch_pop, acc_pop, data_loader, structure)
    t2 = time.time()
    logger.info('Finished, search time is: {} hour.'.format((t2 - t1) / 3600))
