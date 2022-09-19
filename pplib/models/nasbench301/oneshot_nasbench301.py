# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from pplib.nas.mutables import OneShotOP
from ..registry import register_model
from .darts_ops import (DilConv, DropPath, FactorizedReduce, PoolBN, SepConv,
                        StdConv)


class Node(nn.Module):
    """_summary_

    Args:
        node_id (str): _description_
        num_prev_nodes (int): _description_
        channels (int): _description_
        num_downsample_connect (int): _description_
    """

    def __init__(self, node_id: str, num_prev_nodes: int, channels: int,
                 num_downsample_connect: int):

        super().__init__()
        self.ops = nn.ModuleList()
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            candidate_ops = nn.ModuleDict({
                'max_pool_3x3':
                PoolBN('max', channels, 3, stride, 1, affine=False),
                'avg_pool_3x3':
                PoolBN('avg', channels, 3, stride, 1, affine=False),
                'skip_connect':
                nn.Identity() if stride == 1 else FactorizedReduce(
                    channels, channels, affine=False),
                'sep_conv_3x3':
                SepConv(channels, channels, 3, stride, 1, affine=False),
                'sep_conv_5x5':
                SepConv(channels, channels, 5, stride, 2, affine=False),
                'dil_conv_3x3':
                DilConv(channels, channels, 3, stride, 2, 2, affine=False),
                'dil_conv_5x5':
                DilConv(channels, channels, 5, stride, 4, 2, affine=False)
            })

            self.ops.append(
                OneShotOP(
                    candidate_ops=candidate_ops, alias=f'{node_id}_p{i}'))
        self.drop_path = DropPath()

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = []
        for op, node in zip(self.ops, prev_nodes):
            _out = op(node)
            out.append(_out)
        out = [self.drop_path(o) if o is not None else None for o in out]
        return sum(out)


class DartsCell(nn.Module):
    """
    Builtin Darts Cell structure. There are ``n_nodes`` nodes in one cell,
    in which the first two nodes' values are fixed to the results of previous
    previous cell and previous cell respectively. One node will connect all
    the nodes after with predefined operations in a mutable way. The last node
    accepts five inputs from nodes before and it concats all inputs in channels
    as the output of the current cell, and the number of output
    channels is ``n_nodes`` times ``channels``.

    """

    def __init__(
        self,
        n_nodes: int,
        channels_pp: int,
        channels_p: int,
        channels: int,
        reduction_p: bool = False,
        reduction: bool = False,
    ):

        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(
                channels_pp, channels, affine=False)
        else:
            self.preproc0 = StdConv(
                channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(
                Node(f"{'reduce' if reduction else 'normal'}_n{depth}", depth,
                     channels, 2 if reduction else 0))

    def forward(self, pprev, prev):
        tensors = [self.preproc0(pprev), self.preproc1(prev)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        return torch.cat(tensors[2:], dim=1)


@register_model
class OneShotNASBench301Network(nn.Module):
    """
    builtin Darts Search Mutable
    Compared to Darts example, DartsSearchSpace removes Auxiliary Head, which
    is considered as a trick rather than part of model.

    Args:
        in_channels (int, optional): _description_. Defaults to 3.
        channels (int, optional): _description_. Defaults to 16.
        num_classes (int, optional): _description_. Defaults to 10.
        n_layers (int, optional): _description_. Defaults to 8.
        factory_func (_type_, optional): _description_. Defaults to DartsCell.
        n_nodes (int, optional): _description_. Defaults to 4.
        stem_multiplier (int, optional): _description_. Defaults to 3.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 16,
                 num_classes: int = 10,
                 n_layers: int = 8,
                 factory_func=DartsCell,
                 n_nodes: int = 4,
                 stem_multiplier: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.n_layers = n_layers

        c_cur = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_cur))

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size,
        # but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = factory_func(n_nodes, channels_pp, channels_p, c_cur,
                                reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        return self.linear(out)

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, DropPath):
                module.p = p
