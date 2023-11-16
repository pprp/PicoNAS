import numpy as np
import torch
import torch.nn as nn

from piconas.autozc import unary_operation
from piconas.autozc.unary_ops import to_mean_scalar
from . import measure


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


@measure('eznas-a')
def compute_eznas_a(
        net,
        inputs,
        targets,
        loss_fn=nn.CrossEntropyLoss(),
        split_data=1,
):

    def compute_t3g_gradient(net, inputs, targets, loss_fn, split_data=1):
        t3g_list = []  # before relu

        def hook_bw_t3g_fn(module, grad_input, grad_output):
            t3g_list.append(grad_input[1].detach())

        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_backward_hook(hook_bw_t3g_fn)

        logits = net(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss_fn(logits, targets).backward()
        return t3g_list

    inputs = torch.rand_like(inputs)

    t3_list = compute_t3g_gradient(net, inputs, targets, nn.CrossEntropyLoss())

    op_geno = ['element_wise_sign', 'slogdet', 'sigmoid', 'frobenius_norm']

    A = t3_list
    for i, op in enumerate(op_geno):
        A = [unary_operation(a, op) for a in A]
    return convert_to_float(A)
