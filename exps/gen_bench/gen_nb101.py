import itertools
import json
import os
import pdb
import random

import numpy as np
import torch
from nasbench import api
from nasbench_pytorch.model import Network as NBNetwork
from torchvision import datasets, transforms
from tqdm import tqdm

from piconas.predictor.pruners.measures.fisher import (
    compute_fisher_per_weight as compute_fisher,
)
from piconas.predictor.pruners.measures.grad_norm import (
    get_grad_norm_arr as compute_grad_norm,
)
from piconas.predictor.pruners.measures.grasp import (
    compute_grasp_per_weight as compute_grasp,
)
from piconas.predictor.pruners.measures.l2_norm import (
    get_l2_norm_array as compute_l2_norm,
)
from piconas.predictor.pruners.measures.plain import (
    compute_plain_per_weight as compute_plain,
)
from piconas.predictor.pruners.measures.snip import (
    compute_snip_per_weight as compute_snip,
)
from piconas.predictor.pruners.measures.synflow import (
    compute_synflow_per_weight as compute_synflow,
)

BASE = '/data2/dongpeijie/share/bench/predictor_embeddings/embedding_datasets'

# full: 423624
# source_json_path = os.path.join(BASE, 'zc_nasbench101_full.json')  # 9781
# zc_nb101 = json.load(open(source_json_path, 'r'))['cifar10']
# key is (0, 1, 0, 0, ....  0, 0, 3, 4, 3, 4, 4, 1)
# key is dict_keys(['id', 'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov',
#      'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen', 'val_accuracy'])
target_json_path = os.path.join(BASE, 'zc_nasbench101_layerwise.json')
# 'cifar10'
#     #sample '100'
#         index (not hash)
#              'fisher_layerwise': [0.1, 0.2, 0.3, 0.4, 0.5, ...]
to_be_merged_json_path = os.path.join(
    BASE, 'zc_nasbench101_layerwise_5000.json')
to_be_merged_json = json.load(open(to_be_merged_json_path, 'r'))

train_split_list = [100, 172, 424, 424, 4236]

sample_range = np.load('/data/lujunl/pprp/bench/nasbench101/train_samples.npz')


def get_cifar10_dataloader(batch_size, data_dir, train=True):
    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize image tensors
        ]
    )

    # Select the appropriate dataset (train or test)
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=4
    )
    return dataloader


loader = get_cifar10_dataloader(
    128, '/data2/dongpeijie/share/dataset/', train=True)

# get one batch from loader
inputs, targets = next(iter(loader))
if torch.cuda.is_available():
    inputs, targets = inputs.cuda(), targets.cuda()

loss_fn = torch.nn.CrossEntropyLoss()

# zc name candidates
zc_candidates = ['fisher', 'grad_norm', 'grasp',
                 'l2_norm', 'plain', 'snip', 'synflow']

# Build nasbench101 API
nasbench_path = os.path.join(BASE, 'nasbench_only108.tfrecord')
nb = api.NASBench(nasbench_path)
net_hash = nb.hash_iterator()


def min_max_scaling(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)


target_json = dict()
target_json['cifar10'] = dict()


def convert_type(s_list):
    # check s in s_list whether is tensor rather than float
    tmp_list = []
    for s in s_list:
        if isinstance(s, torch.Tensor):
            tmp_list.append(torch.mean(s).item())
        elif isinstance(s, float):
            tmp_list.append(s)
        else:
            print(f'type is: {type(s)}')
            pdb.set_trace()
    return tmp_list


hash_iterator_list = list(nb.hash_iterator())

for split_num in train_split_list:
    index_list = sample_range[str(split_num)].tolist()
    # main iteration
    target_json['cifar10'][str(split_num)] = dict()
    for _idx in tqdm(index_list):
        # convert _idx to _hash
        _hash = hash_iterator_list[_idx]

        target_json['cifar10'][str(split_num)][str(_idx)] = dict()
        # if _hash in to_be_merged_json['cifar10']:
        #     target_json['cifar10'][str(split_num)] = to_be_merged_json['cifar10'][_hash]
        #     continue

        m = nb.get_metrics_from_hash(_hash)
        ops = m[0]['module_operations']
        adjacency = m[0]['module_adjacency']

        methods = {
            'fisher_layerwise': lambda: compute_fisher(
                net, inputs, targets, loss_fn=loss_fn, split_data=1, mode='channel'
            ),
            'grad_norm_layerwise': lambda: compute_grad_norm(
                net, inputs, targets, loss_fn=loss_fn, split_data=1
            ),
            'grasp_layerwise': lambda: compute_grasp(
                net,
                inputs,
                targets,
                loss_fn=loss_fn,
                split_data=1,
                mode='param',
                num_iters=1,
                T=1,
            ),
            'l2_norm_layerwise': lambda: compute_l2_norm(
                net, inputs, targets, loss_fn=loss_fn, split_data=1, mode='param'
            ),
            'plain_layerwise': lambda: compute_plain(
                net, inputs, targets, loss_fn=loss_fn, split_data=1, mode='param'
            ),
            'snip_layerwise': lambda: compute_snip(
                net, inputs, targets, loss_fn=loss_fn, split_data=1, mode='param'
            ),
            'synflow_layerwise': lambda: compute_synflow(
                net, inputs, targets, loss_fn=loss_fn, split_data=1, mode='param'
            ),
        }

        for method in methods:
            net = NBNetwork((adjacency, ops))
            if torch.cuda.is_available():
                net = net.cuda()
            s_list = methods[method]()
            tmp_list = convert_type(s_list)
            tmp_list = min_max_scaling(tmp_list)
            target_json['cifar10'][str(split_num)][str(_idx)][
                method
            ] = tmp_list.tolist()
            del net  # release memory

        # print(target_json['cifar10'][str(split_num)].keys())

    # print(target_json['cifar10'][str(split_num)].keys())

# save target_json
with open(target_json_path, 'w') as f:
    json.dump(target_json, f)
