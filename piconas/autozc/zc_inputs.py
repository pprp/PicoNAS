import copy
from typing import List

import autozc.utils.autograd_hacks as autograd_hacks
import numpy as np
import torch
import torch.nn as nn
from autozc.predictor.pruners.measures.linear_region import \
    Linear_Region_Collector
from autozc.predictor.pruners.predictive import get_layer_metric_array
from autozc.utils.autograd_hacks import clear_backprops
from torch import Tensor

from . import zc_candidates

# @zc_candidates('logits')
# def compute_logits_activation(net,
#                               inputs,
#                               targets,
#                               loss_fn,
#                               split_data=1) -> Tensor:
#     return net(inputs).detach()


@zc_candidates('t1')
def compute_t1_activation(net, inputs, targets, loss_fn, split_data=1) -> List:
    t1_list = []  # before relu

    def hook_fw_t1_fn(module, input, output):
        t1_list.append(input[0].detach())

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
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(hook_fw_t1_fn)

    _ = net(inputs)
    return t1_list


@zc_candidates('t1g')
def compute_t1g_gradient(net, inputs, targets, loss_fn, split_data=1) -> List:
    t1g_list = []  # before relu

    def hook_bw_t1g_fn(module, grad_input, grad_output):
        t1g_list.append(grad_input[0].detach())

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
        if isinstance(module, nn.ReLU):
            module.register_backward_hook(hook_bw_t1g_fn)

    logits = net(inputs)
    loss_fn(logits, targets).backward()
    return t1g_list


@zc_candidates('t2')
def compute_t2_activation(net, inputs, targets, loss_fn, split_data=1) -> List:
    t2_list = []  # before relu

    def hook_fw_t2_fn(module, input, output):
        t2_list.append(input[0].detach())

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
            module.register_forward_hook(hook_fw_t2_fn)

    _ = net(inputs)
    return t2_list


@zc_candidates('t2g')
def compute_t2g_gradient(net, inputs, targets, loss_fn, split_data=1) -> List:
    t2g_list = []  # before relu

    def hook_bw_t2g_fn(module, grad_input, grad_output):
        t2g_list.append(grad_input[0].detach())

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
            module.register_backward_hook(hook_bw_t2g_fn)

    logits = net(inputs)
    loss_fn(logits, targets).backward()
    return t2g_list


@zc_candidates('t3')
def compute_t3_activation(net, inputs, targets, loss_fn, split_data=1) -> List:
    t3_list = []  # before relu

    def hook_fw_t3_fn(module, input, output):
        t3_list.append(module.weight.detach())

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
            module.register_forward_hook(hook_fw_t3_fn)

    _ = net(inputs)
    return t3_list


@zc_candidates('t3g')
def compute_t3g_gradient(net, inputs, targets, loss_fn, split_data=1) -> List:
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


@zc_candidates('t4')
def compute_t4_activation(net, inputs, targets, loss_fn, split_data=1) -> List:
    t4_list = []  # before relu

    def hook_fw_t4_fn(module, input, output):
        t4_list.append(output.detach())

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
        if isinstance(module, nn.BatchNorm2d):
            module.register_forward_hook(hook_fw_t4_fn)

    _ = net(inputs)
    return t4_list


@zc_candidates('t4g')
def compute_t4g_gradient(net, inputs, targets, loss_fn, split_data=1) -> List:
    t4g_list = []  # before relu

    def hook_bw_t4g_fn(module, grad_input, grad_output):
        t4g_list.append(grad_output[0].detach())

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
        if isinstance(module, nn.BatchNorm2d):
            module.register_backward_hook(hook_bw_t4g_fn)

    logits = net(inputs)
    loss_fn(logits, targets).backward()
    return t4g_list


# @zc_candidates('grad_vector')
# def compute_gradient_vector(net,
#                             inputs,
#                             targets,
#                             loss_fn,
#                             split_data=1) -> List:
#     net.zero_grad()
#     outputs = net(inputs)
#     loss = loss_fn(outputs, targets)
#     loss.backward()

#     grad_list = []
#     for name, layer in net.named_modules():
#         if 'stem' in name:
#             continue
#         if 'lastact' in name:
#             continue
#         # two downsample layers
#         if 'cells.5' in name:
#             continue
#         if 'cells.11' in name:
#             continue

#         if isinstance(layer, nn.Conv2d):
#             if layer.weight.grad is not None:
#                 grad_list.append(layer.weight.grad.detach())

#     # vectorize
#     grad_vector = []
#     for grad_m in grad_list:
#         grad_v = grad_m.reshape(-1)
#         grad_vector.append(grad_v)
#     return grad_vector

# @zc_candidates('all1gradient')
# def compute_all1_gradient(net, inputs, targets, loss_fn, split_data=1) -> List:
#     '''synflow-like'''
#     net.zero_grad()

#     input_dim = list(inputs[0, :].shape)
#     input = torch.ones([1] + input_dim)
#     input = input.to(inputs.device)

#     output = net(input)
#     torch.sum(output).backward()

#     all1_grad = []
#     for name, layer in net.named_modules():
#         if 'stem' in name:
#             continue
#         if 'lastact' in name:
#             continue
#         # two downsample layers
#         if 'cells.5' in name:
#             continue
#         if 'cells.11' in name:
#             continue

#         if isinstance(layer, nn.Conv2d):
#             if layer.weight.grad is not None:
#                 all1_grad.append(layer.weight.grad.detach())

#     return all1_grad

# @zc_candidates('hessian')
# def compute_hessian_gradient(net,
#                              inputs,
#                              targets,
#                              loss_fn,
#                              split_data=1) -> tuple:
#     net.zero_grad()
#     prob = 0.01
#     device = inputs.device

#     output = net(inputs)
#     loss = loss_fn(output, targets)

#     params = []
#     for name, param in net.named_parameters():
#         if 'stem' in name:
#             continue
#         if 'lastact' in name:
#             continue
#         # two downsample layers
#         if 'cells.5' in name:
#             continue
#         if 'cells.11' in name:
#             continue
#         if param.requires_grad and 'op.1.weight' in name:
#             params.append(param)

#     grads = torch.autograd.grad(
#         loss, params, retain_graph=True, create_graph=True)

#     grad_list = []
#     for i in grads:
#         grad_list.append(i)

#     with torch.no_grad():
#         v = [torch.randint_like(p, high=1, device=device) for p in params]
#         for v_i in v:
#             v_i[v_i == 0] = np.random.binomial(1, prob * 2)
#         for v_i in v:
#             v_i[v_i == 1] = 2 * np.random.binomial(1, 0.5) - 1

#     hessian = torch.autograd.grad(
#         grad_list, params, grad_outputs=v, only_inputs=True, retain_graph=True)
#     return hessian

# @zc_candidates('gram')
# def compute_gram_gradient(net,
#                           inputs,
#                           targets,
#                           loss_fn,
#                           split_data=1) -> Tensor:
#     net.zero_grad()
#     outputs = net(inputs)
#     loss = loss_fn(outputs, targets)
#     loss.backward()

#     index = 0
#     grad_dict = {}
#     g = 0
#     para = 0
#     for name, param in net.named_parameters():
#         if param.grad is None:
#             continue

#         if index > 10:
#             break
#         index += 1

#         if len(param.grad.view(-1).data[:100]) < 50:
#             continue

#         if name in grad_dict:
#             grad_dict[name].append(copy.copy(param.grad.view(-1).data[:100]))
#         else:
#             grad_dict[name] = [copy.copy(param.grad.view(-1).data[:100])]

#     for name in grad_dict:
#         for i in range(len(grad_dict[name])):
#             grad1 = torch.tensor([grad_dict[name][i][k] for k in range(25)])
#             grad2 = torch.tensor(
#                 [grad_dict[name][i][k] for k in range(25, 50)])
#             grad1 = grad1 - grad1.mean()
#             grad2 = grad2 - grad2.mean()
#             g += torch.dot(grad1, grad2) / 2500
#             para += 1
#     return torch.tensor(g.item() / para).detach()

# @zc_candidates('jacobian')
# def compute_jacobian_gradient(net,
#                               inputs,
#                               targets,
#                               loss_fn,
#                               split_data=1) -> Tensor:
#     net.zero_grad()
#     inputs.requires_grad_(True)
#     y = net(inputs)

#     def _random_vector(C, B):
#         '''
#         creates a random vector of dimension C with a norm of C^(1/2)
#         (as needed for the projection formula to work)
#         '''
#         if C == 1:
#             return torch.ones(B)
#         v = torch.randn(B, C)
#         arxilirary_zero = torch.zeros(B, C)
#         vnorm = torch.norm(v, 2, 1, True)
#         v = torch.addcdiv(arxilirary_zero, v, vnorm, value=1.0)
#         return v

#     def _jacobian_vector_product(y, x, v, create_graph=False):
#         '''
#         Produce jacobian-vector product dy/dx dot v.
#         Note that if you want to differentiate it,
#         you need to make create_graph=True
#         '''
#         flat_y = y.reshape(-1)
#         flat_v = v.reshape(-1)
#         grad_x, = torch.autograd.grad(
#             flat_y, x, flat_v, retain_graph=True, create_graph=create_graph)
#         return grad_x

#     B, C = y.shape
#     # random properly-normalized vector for each sample
#     v = _random_vector(C=C, B=B)
#     if inputs.is_cuda:
#         v = v.cuda()
#     Jv = _jacobian_vector_product(y, inputs, v, create_graph=True)

#     return Jv

# @zc_candidates('ntk')
# def compute_ntk(net, inputs, targets, loss_fn, split_data=1) -> Tensor:
#     net.zero_grad()
#     N = inputs.shape[0]

#     clear_backprops(net)
#     autograd_hacks.add_hooks(net)
#     outputs = net.forward(inputs)
#     sum(outputs[torch.arange(N), targets]).backward()
#     autograd_hacks.compute_grad1(net, loss_type='sum')

#     grads = [
#         param.grad1.flatten(start_dim=1) for param in net.parameters()
#         if hasattr(param, 'grad1')
#     ]
#     grads = torch.cat(grads, axis=1)

#     return torch.matmul(grads, grads.t()).detach()

# @zc_candidates('nlr')
# def compute_number_linear_region(net,
#                                  inputs,
#                                  targets,
#                                  loss_fn,
#                                  split_data=1) -> float:
#     lrc_model = Linear_Region_Collector(
#         models=[net],
#         input_size=inputs.shape,
#         gpu=None,
#         sample_batch=1,
#         device=inputs.device)
#     num_linear_regions = float(lrc_model.forward_batch_sample()[0])
#     del lrc_model
#     return torch.tensor(num_linear_regions).detach()
