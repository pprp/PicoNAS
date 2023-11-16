import itertools
import json
import os
import pdb
import random

import numpy as np
import torch
from nas_201_api import NASBench201API as API
from torchvision import datasets, transforms
from tqdm import tqdm

from piconas.datasets.imagenet16 import ImageNet16
from piconas.models.nasbench201.apis.utils import dict2config, get_cell_based_tiny_net
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
# source_json_path = os.path.join(BASE, 'zc_nasbench201_full.json')  # 9781
# zc_nb201 = json.load(open(source_json_path, 'r'))['cifar10']
# key is (0, 1, 0, 0, ....  0, 0, 3, 4, 3, 4, 4, 1)
# key is dict_keys(['id', 'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov',
#      'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen', 'val_accuracy'])
target_json_path = os.path.join(BASE, 'zc_nasbench201_layerwise.json')


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


def get_cifar100_dataloader(batch_size, data_dir, train=True):
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
    dataset = datasets.CIFAR100(
        root=data_dir, train=train, download=True, transform=transform
    )

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=4
    )
    return dataloader


def get_imagenet16_dataloader(batch_size, data_dir, train=True):
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
    dataset = ImageNet16(
        root=data_dir, train=train, transform=transform, use_num_of_class_only=120
    )

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=4
    )
    return dataloader


# get one batch from loader
loss_fn = torch.nn.CrossEntropyLoss()

# zc name candidates
zc_candidates = ['fisher', 'grad_norm', 'grasp',
                 'l2_norm', 'plain', 'snip', 'synflow']

# Build nasbench201 API
nb201_api = API(
    '/data2/dongpeijie/share/bench/NAS-Bench-201-v1_0-e61699.pth', verbose=False
)


def min_max_scaling(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


target_json = dict()
target_json['cifar10'] = dict()
target_json['cifar100'] = dict()
target_json['ImageNet16-120'] = dict()


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


def generate_model(index, NUM_CLASSES=10):
    assert index in list(range(15625))
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(index),
        'num_classes': NUM_CLASSES,
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    return model


dataset_candidates = ['cifar10', 'cifar100']

for data_cand in dataset_candidates:
    if data_cand == 'cifar10':
        loader = get_cifar10_dataloader(
            16, '/data2/dongpeijie/share/dataset/', train=True
        )
        NUM_CLASSES = 10
    elif data_cand == 'cifar100':
        loader = get_cifar100_dataloader(
            16, '/data2/dongpeijie/share/dataset/', train=True
        )
        NUM_CLASSES = 100
    elif data_cand == 'ImageNet16-120':
        loader = get_imagenet16_dataloader(
            16, '/data2/dongpeijie/share/dataset/ImageNet16', train=True
        )
        NUM_CLASSES = 120
    inputs, targets = next(iter(loader))

    # main iteration
    for _index in tqdm(range(15625)):
        target_json[data_cand][str(_index)] = dict()

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
            net = generate_model(_index, NUM_CLASSES)
            s_list = methods[method]()
            tmp_list = convert_type(s_list)
            tmp_list = min_max_scaling(tmp_list)
            target_json[data_cand][str(_index)][method] = tmp_list.tolist()

# save target_json
with open(target_json_path, 'w') as f:
    json.dump(target_json, f)
