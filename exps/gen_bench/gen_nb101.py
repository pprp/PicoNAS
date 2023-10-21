import json
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
from nasbench import api
from nasbench_pytorch.model import Network as NBNetwork
from torchvision import datasets, transforms

from piconas.predictor.pruners.measures.fisher import compute_fisher_per_weight

BASE = '/data2/dongpeijie/share/bench/predictor_embeddings/embedding_datasets'

# full: 423624
source_json_path = os.path.join(BASE, 'zc_nasbench101_full.json')  # 9781
target_json_path = os.path.join(BASE, 'zc_nasbench101_layerwise.json')
zc_nb101 = json.load(open(source_json_path, 'r'))['cifar10']
# key is (0, 1, 0, 0, ....  0, 0, 3, 4, 3, 4, 4, 1)
# key is dict_keys(['id', 'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov',
#      'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen', 'val_accuracy'])


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


loader = get_cifar10_dataloader(
    128, '/data2/dongpeijie/share/dataset/', train=True)

# get one batch from loader
inputs, targets = next(iter(loader))
loss_fn = torch.nn.CrossEntropyLoss()

# zc name candidates
zc_candidates = [
    'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'plain', 'snip',
    'synflow'
]

# Build nasbench101 API
nasbench_path = os.path.join(BASE, 'nasbench_only108.tfrecord')
nb = api.NASBench(nasbench_path)
net_hash = nb.hash_iterator()


def min_max_scaling(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# copy zc_nb101 to target_json
target_json = {}
for key in zc_nb101.keys():
    target_json[key] = {}
    for zc in zc_candidates:
        target_json[key][zc] = zc_nb101[key][zc]

# main iteration
for _hash in net_hash:
    m = nb.get_metrics_from_hash(_hash)
    ops = m[0]['module_operations']
    adjacency = m[0]['module_adjacency']
    net = NBNetwork((adjacency, ops))

    key = tuple(adjacency.flatten().tolist())

    s_list = compute_fisher_per_weight(
        net, inputs, targets, loss_fn, split_data=1, mode='channel')

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

    tmp_list = min_max_scaling(tmp_list)

    # update target_json
    target_json[key]['fisher_layerwise'] = tmp_list

# save target_json
with open(target_json_path, 'w') as f:
    json.dump(target_json, f)
