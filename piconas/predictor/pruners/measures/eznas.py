import torch.nn as nn
from autozc.operators import (available_zc_candidates, binary_operation,
                              get_zc_candidates, sample_binary_key_by_prob,
                              sample_unary_key_by_prob, unary_operation)
from autozc.operators.binary_ops import BINARY_KEYS
from autozc.operators.unary_ops import UNARY_KEYS
from autozc.structures.utils import convert_to_float

from . import measure


@measure('EZNAS-A')
def compute_score1(
        net,
        inputs,
        targets,
        loss_fn=nn.CrossEntropyLoss(),
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
