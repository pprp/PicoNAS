import json

import torch
import torch.nn as nn

from exps.mq_bench_101.resnet18 import resnet18
from piconas.core.api.mq_bench_101 import EMQAPI
from piconas.predictor.pruners.p_utils import get_layer_metric_array

emqapi = EMQAPI('./checkpoints/MQ-Bench-101-PTQ-GT.pkl', verbose=False)


def compute_synflow_per_weight(net, inputs, targets, mode='param'):

    device = inputs.device

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net.double())

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None and isinstance(layer, nn.Conv2d):
            return torch.abs(layer.weight * layer.weight.grad)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs


if __name__ == '__main__':
    save_path = './checkpoints/mq-bench-layerwise-zc.json'

    model = resnet18(num_classes=1000)
    inputs = torch.randn(4, 3, 256, 256)

    data_to_save = []

    for i in range(50):
        bit_cfg = emqapi.fix_bit_cfg(i)
        gt = emqapi.query_by_cfg(bit_cfg)

        layerwise_zc = compute_synflow_per_weight(
            net=model, inputs=inputs, targets=None)[:len(bit_cfg)]

        lw_zc_list = []
        # zip layerwise_zc and bit_cfg
        for j in range(len(layerwise_zc)):
            lw_zc_list.append(torch.mean(layerwise_zc[j]).item() * bit_cfg[j])

        entry = {'id': i + 1, 'layerwise_zc': lw_zc_list, 'gt': gt}

        data_to_save.append(entry)

        print(f'Finish {i + 1} th model')

    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(data_to_save, f)
