import itertools
import json
import os
import pdb
from copy import deepcopy

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from piconas.predictor.pruners.measures.fisher import \
    compute_fisher_per_weight as compute_fisher
from piconas.predictor.pruners.measures.grad_norm import \
    get_grad_norm_arr as compute_grad_norm
from piconas.predictor.pruners.measures.grasp import \
    compute_grasp_per_weight as compute_grasp
from piconas.predictor.pruners.measures.l2_norm import \
    get_l2_norm_array as compute_l2_norm
from piconas.predictor.pruners.measures.plain import \
    compute_plain_per_weight as compute_plain
from piconas.predictor.pruners.measures.snip import \
    compute_snip_per_weight as compute_snip
from piconas.predictor.pruners.measures.synflow import \
    compute_synflow_per_weight as compute_synflow
from piconas.utils.get_dataset_api import NDS

BASE = './checkpoints/nds_data/'

if not os.path.exists(BASE):
    os.makedirs(BASE)


def get_cifar10_dataloader(batch_size, data_dir, train=True):
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))  # Normalize image tensors
    ])

    # Select the appropriate dataset (train or test)
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform)

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if train else False,
        num_workers=4)
    return dataloader


# get one batch from loader
loss_fn = torch.nn.CrossEntropyLoss()

# zc name candidates
zc_candidates = [
    'fisher', 'grad_norm', 'grasp', 'l2_norm', 'plain', 'snip', 'synflow'
]


def min_max_scaling(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


target_json = dict()


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


dataset = 'cifar10'

loader = get_cifar10_dataloader(16, './data/cifar', train=True)
NUM_CLASSES = 10
inputs, targets = next(iter(loader))

search_space = ['Amoeba', 'DARTS', 'ENAS', 'NASNet', 'PNAS']

for ss in search_space:
    target_json[ss] = dict()
    nds_api = NDS(ss)
    iter_nds = iter(nds_api)

    for uid in tqdm(range(len(nds_api))):

        # debug
        if uid > 1000:
            break

        methods = {
            'fisher_layerwise':
            lambda: compute_fisher(
                net,
                inputs,
                targets,
                loss_fn=loss_fn,
                split_data=1,
                mode='channel'),
            'grad_norm_layerwise':
            lambda: compute_grad_norm(
                net, inputs, targets, loss_fn=loss_fn, split_data=1),
            'grasp_layerwise':
            lambda: compute_grasp(
                net,
                inputs,
                targets,
                loss_fn=loss_fn,
                split_data=1,
                mode='param',
                num_iters=1,
                T=1),
            'l2_norm_layerwise':
            lambda: compute_l2_norm(
                net,
                inputs,
                targets,
                loss_fn=loss_fn,
                split_data=1,
                mode='param'),
            'plain_layerwise':
            lambda: compute_plain(
                net,
                inputs,
                targets,
                loss_fn=loss_fn,
                split_data=1,
                mode='param'),
            'snip_layerwise':
            lambda: compute_snip(
                net,
                inputs,
                targets,
                loss_fn=loss_fn,
                split_data=1,
                mode='param'),
            'synflow_layerwise':
            lambda: compute_synflow(
                net,
                inputs,
                targets,
                loss_fn=loss_fn,
                split_data=1,
                mode='param')
        }

        target_json[ss][uid] = dict()
        target_json[ss]['gt'] = nds_api.get_final_accuracy(uid)

        for method in methods:
            net = nds_api[uid]
            s_list = methods[method]()
            tmp_list = convert_type(s_list)
            tmp_list = min_max_scaling(tmp_list)
            target_json[ss][uid][method] = tmp_list.tolist()

target_json_path = os.path.join(BASE, f'nds_ss_layerwise_zc.json')
# save target_json
with open(target_json_path, 'w') as f:
    json.dump(target_json, f)
