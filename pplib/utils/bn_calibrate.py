import torch.nn as nn


def separate_bn_params(model):
    bn_index = []
    bn_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_index += list(map(id, m.parameters()))
            bn_params += m.parameters()
    base_params = list(
        filter(lambda p: id(p) not in bn_index, model.parameters()))
    return base_params, bn_params
