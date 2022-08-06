from typing import Dict, List

import torch
import torch.nn as nn

from pplib.models.spos import SearchableMobileNet
from pplib.nas.mutators import OneShotMutator


def get_mb_arch_vector(supernet: SearchableMobileNet,
                       mutator: OneShotMutator,
                       subnet_dict: Dict = None) -> List:
    """get SearchableMobileNet arch vector

    Args:
        supernet (nn.Module): searchable mobilenet supernet.
        mutator (OneShotMutator): oneshotmutator for searchable mobilenet.
    """

    first_conv = torch.reshape(supernet.first_conv[0].weight.data, (-1, ))
    conv1 = torch.reshape(supernet.stem_MBConv.conv[0].weight.data, (-1, ))
    conv2 = torch.reshape(supernet.stem_MBConv.conv[3].weight.data, (-1, ))

    last_conv = torch.reshape(supernet.last_conv[0].weight.data, (-1, ))
    classifier = torch.reshape(supernet.classifier.weight.data, (-1, ))

    arch_vector = [first_conv, conv1, conv2, last_conv, classifier]

    if subnet_dict is None:
        subnet_dict = mutator.random_subnet
    else:
        mutator.set_subnet(subnet_dict)

    for group_id, modules in mutator.search_group.items():
        choice = subnet_dict[group_id]
        # print(choice, len(modules), type(modules[0]))
        # print(modules[0]._candidate_ops.keys())
        conv1 = torch.reshape(
            modules[0]._candidate_ops[choice].conv[0].weight.data, (-1, ))
        conv2 = torch.reshape(
            modules[0]._candidate_ops[choice].conv[3].weight.data, (-1, ))
        conv3 = torch.reshape(
            modules[0]._candidate_ops[choice].conv[6].weight.data, (-1, ))
        arch_vector += [torch.cat([conv1, conv2, conv3], dim=0)]

    arch_vector = torch.cat(arch_vector, dim=0)
    return arch_vector


def get_mb_angle(
    base_model: SearchableMobileNet,
    base_mutator: OneShotMutator,
    model: SearchableMobileNet,
    mutator: OneShotMutator,
    subnet_dict: Dict,
):
    cosine = nn.CosineSimilarity(dim=0)
    vec1 = get_mb_arch_vector(base_model, base_mutator, subnet_dict)
    vec2 = get_mb_arch_vector(model, mutator, subnet_dict)
    angle = torch.acos(cosine(vec1, vec2))
    return angle
