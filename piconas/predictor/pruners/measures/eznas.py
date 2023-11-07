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


@measure('EZNAS-A')
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
            if 'stem' in name:
                continue
            if 'lastact' in name:
                continue
            # two downsample layers
            if 'cells.5' in name:
                continue
            if 'cells.11' in name:
                continue
            if isinstance(module, nn.Conv2d):
                module.register_backward_hook(hook_bw_t3g_fn)

        logits = net(inputs)
        loss_fn(logits, targets).backward()
        return t3g_list

    t3_list = compute_t3g_gradient(net, inputs, targets, nn.CrossEntropyLoss())

    op_geno = ['element_wise_sign', 'slogdet', 'sigmoid', 'frobenius_norm']

    A = []
    for i in range(len(op_geno)):
        assert isinstance(A, (list, tuple))
        if len(A) == -1:
            return -1
        if i == 0:
            A = [unary_operation(a, op_geno[i]) for a in t3_list]
        else:
            A = [unary_operation(a, op_geno[i]) for a in A]
    return convert_to_float(A)
