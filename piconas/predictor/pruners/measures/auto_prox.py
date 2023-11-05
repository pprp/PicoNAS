# auto_prox A
# Input=['t3g', 't2'] Op=[[5, 10], [14, 6], 0]
# auto_prox P
# Input=['t2', 't1g'] Op=[[8, 6], [10, 1], 1]

from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn as nn

from piconas.autozc import binary_operation, unary_operation
from piconas.autozc.binary_ops import BINARY_KEYS
from piconas.autozc.unary_ops import UNARY_KEYS, to_mean_scalar
from . import measure  # noqa: F401


def convert_to_float(input):
    if isinstance(input, (list, tuple)):
        if len(input) == 0:
            return -1
        return sum(convert_to_float(x) for x in input) / len(input)
    elif isinstance(input, torch.Tensor):
        return to_mean_scalar(input).item()
    elif isinstance(input, np.ndarray):
        return input.astype(float)
    elif isinstance(input, (int, float)):
        return input
    else:
        print(type(input))
        return float(input)


@measure('auto-prox-A')
def compute_score1(
        net,
        inputs,
        targets,
        loss_fn=nn.CrossEntropyLoss(),
        split_data=None,
):

    def compute_t3g(net, inputs, targets, loss_fn) -> List:
        t3g_list = []

        def hook_bw_t3g_fn(module, grad_input, grad_output):
            if grad_input is not None and grad_input[0] is not None:
                t3g_list.append(grad_input[0].detach())
            else:
                print('Warning: grad_input is None or grad_input[0] is None')

        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module.register_backward_hook(hook_bw_t3g_fn)

        logits = net(inputs)
        loss_fn(logits, targets).backward()
        return t3g_list

    def compute_t2(net, inputs) -> List:
        t2_list = []  # before relu

        def hook_fw_t2_fn(module, input, output):
            t2_list.append(module.weight.detach())

        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fw_t2_fn)

        _ = net(inputs)
        return t2_list

    net1 = deepcopy(net)
    t3_list = compute_t3g(net1, inputs, targets, loss_fn)
    A1 = [unary_operation(a, UNARY_KEYS[5]) for a in t3_list]
    A1 = [unary_operation(a, UNARY_KEYS[10]) for a in A1]

    net2 = deepcopy(net)
    t2_list = compute_t2(net2, inputs)
    A2 = [unary_operation(a, UNARY_KEYS[14]) for a in t2_list]
    A2 = [unary_operation(a, UNARY_KEYS[6]) for a in A2]

    A = []
    for a1, a2 in zip(A1, A2):
        a1 = convert_to_float(a1)
        a2 = convert_to_float(a2)
        A.append(binary_operation(a1, a2, BINARY_KEYS[0]))
    return convert_to_float(A)


@measure('auto-prox-P')
def compute_score2(
        net,
        inputs,
        targets,
        loss_fn=nn.CrossEntropyLoss(),
        split_data=None,
):

    def compute_t2(net, inputs) -> List:
        t2_list = []  # before relu

        def hook_fw_t2_fn(module, input, output):
            t2_list.append(module.weight.detach())

        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fw_t2_fn)

        _ = net(inputs)
        return t2_list

    def compute_t1g(net, inputs, targets, loss_fn) -> List:
        t1g_list = []  # before relu

        def hook_bw_t1g_fn(module, grad_input, grad_output):
            if grad_input is not None and grad_input[0] is not None:
                t1g_list.append(grad_input[0].detach())

        for name, module in net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(hook_bw_t1g_fn)

        logits = net(inputs)
        loss_fn(logits, targets).backward()
        return t1g_list

    t2_list = compute_t2(net, inputs)
    A1 = [unary_operation(a, UNARY_KEYS[8]) for a in t2_list]
    A1 = [unary_operation(a, UNARY_KEYS[6]) for a in A1]

    t1_list = compute_t1g(net, inputs, targets, nn.CrossEntropyLoss())
    A2 = [unary_operation(a, UNARY_KEYS[10]) for a in t1_list]
    A2 = [unary_operation(a, UNARY_KEYS[1]) for a in A2]

    A = []
    for a1, a2 in zip(A1, A2):
        a1 = convert_to_float(a1)
        a2 = convert_to_float(a2)
        A.append(binary_operation(a1, a2, BINARY_KEYS[1]))
    return convert_to_float(A)
