import numpy as np
import torch
from torch import nn

from . import measure


def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if mod.weight.grad is None:
                    print(f'Warning: {name} grad is None. {mod}')
                grad_dict[name] = [
                    mod.weight.grad.data.cpu().reshape(-1).numpy()]
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                grad_dict[name].append(
                    mod.weight.grad.data.cpu().reshape(-1).numpy())
    return grad_dict


def caculate_zico(grad_dict):
    allgrad_array = None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(
                np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
            )
    return nsr_mean_sum_abs


@measure('zico_ori')
def getzico(network, trainloader, loss_fn=None, split_data=1):
    grad_dict = {}
    network.train()

    network.cuda()
    for i, batch in enumerate(trainloader):
        if i > 20:
            break
        network.zero_grad()
        data, label = batch[0], batch[1]
        data, label = data.cuda(), label.cuda()

        logits = network(data)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = loss_fn(logits, label)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        try:
            grad_dict = getgrad(network, grad_dict, i)
        except:
            pass

    res = caculate_zico(grad_dict)
    return res


@measure('zico')
def compute_zico_score(net, inputs, targets, loss_fn=None, split_data=1):
    grad_dict = {}
    net.train()
    for i in range(split_data):
        net.zero_grad()
        logits = net(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = loss_fn(logits, targets)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        grad_dict = getgrad(net, grad_dict, i)
    res = caculate_zico(grad_dict)
    return res
