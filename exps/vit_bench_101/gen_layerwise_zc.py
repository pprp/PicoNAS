import json
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exps.vit_bench_101.autoformer import AutoFormerSub
from exps.vit_bench_101.dataset import Cifar100
from piconas.core.api.vit_bench_101 import ViTBenchAPI
from piconas.predictor.pruners.p_utils import get_layer_metric_array

vitapi = ViTBenchAPI('checkpoints/af_100.pth')


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def compute_snip_per_weight(
    net, inputs, targets, mode='param', loss_fn=nn.CrossEntropyLoss(), split_data=1
):
    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # split whole batch into split_data parts.
        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, snip, mode)

    return grads_abs


if __name__ == '__main__':
    save_path = './checkpoints/vit-bench-layerwise-zc.json'

    dataset = Cifar100(data_path='./data/cifar', split='train')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    inputs, targets, _ = next(iter(dataloader))

    data_to_save = []

    for i in range(len(vitapi)):
        arch_cfg = vitapi.query_by_idx(i)['arch']
        gt_base = vitapi.query_by_idx(i)['cifar100']['base']

        model = AutoFormerSub(arch_cfg, num_classes=100)

        layerwise_zc = compute_snip_per_weight(model, inputs, targets)

        lw_zc_list = []
        # zip layerwise_zc and bit_cfg
        for j in range(len(layerwise_zc)):
            lw_zc_list.append(torch.mean(layerwise_zc[j]).item())

        # apply min-max scale to layerwise_zc
        lw_zc_list = np.array(lw_zc_list)
        lw_zc_list = (lw_zc_list - min(lw_zc_list)) / (
            max(lw_zc_list) - min(lw_zc_list)
        )
        lw_zc_list = lw_zc_list.tolist()

        entry = {'id': i + 1, 'layerwise_zc': lw_zc_list, 'gt': gt_base}

        data_to_save.append(entry)

        print(f'Finish {i + 1} th model')

    # pad to max length
    max_len = 0
    for item in data_to_save:
        max_len = max(max_len, len(item['layerwise_zc']))
    for item in data_to_save:
        item['layerwise_zc'] = item['layerwise_zc'] + [0] * (
            max_len - len(item['layerwise_zc'])
        )
    print(max_len)

    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(data_to_save, f)
