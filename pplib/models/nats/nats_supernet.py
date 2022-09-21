"""
License: https://github.com/changlin31/BossNAS
"""

import numpy as np
import torch.nn as nn

from pplib.models.nats.nats_ops import (InferCell, ResNetBasicblock,
                                        SlimmableConv2d, SlimmableLinear)
from pplib.models.nats.nats_ops import Structure as CellStructure
from pplib.models.nats.nats_ops import SwitchableBatchNorm2d
from pplib.models.nats.utils import reset
from ..registry import register_model


def uniform_random_op_encoding(num_of_ops, layers):
    return np.random.randint(0, num_of_ops, layers)


def fair_random_op_encoding(num_of_ops, layers):
    # return alist
    encodings = np.zeros((layers, num_of_ops), dtype=np.int8)
    for i in range(layers):
        encodings[:][i] = np.random.choice(
            np.arange(0, num_of_ops), size=num_of_ops, replace=False)
    return encodings.T.tolist()


def get_path(str, num):
    if num == 1:
        yield from str
    else:
        for x in str:
            for y in get_path(str, num - 1):
                yield x + y


def all_op_encoding(num_of_ops, layers):
    strKey = ''.join(str(x) for x in range(num_of_ops))
    return [[int(op) for op in path] for path in get_path(strKey, layers)]


class MixOps(nn.Module):
    """Each mixops has 6 operations"""

    def __init__(self, reduction=False):
        super(MixOps, self).__init__()
        genotype = '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'
        self.genotype = CellStructure.str2structure(genotype)
        self.candidate_Cs = [8, 16, 24, 32, 40, 48, 56, 64]
        self._max_C = max(self.candidate_Cs)
        if reduction:
            self._mix_ops = ResNetBasicblock(self._max_C, self._max_C, 2, True)
        else:
            self._mix_ops = InferCell(
                self.genotype,
                self._max_C,
                self._max_C,
                1,
                affine=False,
                track_running_stats=True,
            )

    def forward(self, x, in_idx, out_idx):
        # Single-path
        return self._mix_ops(x, in_idx, out_idx)


class Block(nn.Module):
    """Each block has n layers"""

    def __init__(self, layers, stage, target):
        super(Block, self).__init__()
        self._block_layers = nn.ModuleList()

        if layers == 1:
            self._block_layers.append(MixOps(reduction=False))
        else:
            for i in range(layers):
                if target == 'cifar10':
                    if i == 0:
                        self._block_layers.append(MixOps(reduction=False))
                    elif i == layers - 1:
                        self._block_layers.append(MixOps(reduction=True))
                elif target == 'cifar100':
                    if i == 0:
                        self._block_layers.append(MixOps(reduction=stage != 0))
                    elif i == layers - 1:
                        self._block_layers.append(MixOps(reduction=stage == 0))

    def forward(self, x, start_block, forward_list=None, pre_op=None):
        """

        Args:
            start_block (_type_): judge wether is the first block
            pre_op (_type_, optional): the last channel of last block
        """
        # print(f'In block: forwad_list={forward_list}')
        assert len(forward_list) == len(self._block_layers)
        for i, layer in enumerate(self._block_layers):
            out_idx = forward_list[i]
            if i == 0:
                # process the last channel from last block
                if start_block == 0:
                    in_idx = out_idx
                else:
                    assert pre_op is not None
                    in_idx = pre_op
            else:
                in_idx = forward_list[i - 1]
            # print(f' == > layer_{i} in_idx: {in_idx} out_idx: {out_idx}')
            x = layer(x, in_idx=in_idx, out_idx=out_idx)
        return x

    def reset_params(self):
        self.apply(reset)


@register_model
class SupernetNATS(nn.Module):
    """Three Blocks, with [2, 2, 1] layers"""

    def __init__(self, target='cifar10', num_classes=10):
        super(SupernetNATS, self).__init__()
        self.max_num_Cs = 5
        self.candidate_Cs = [8, 16, 24, 32, 40, 48, 56, 64]
        self.FLAGS = {
            'width_mult_list':
            [0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
        }
        self._num_of_ops = 8
        self.target = target

        # number of layers per block
        self._op_layers_list = [2, 2, 1] if target == 'cifar10' else [2, 1, 2]

        # number of class
        self._num_classes = num_classes

        # generate channel list
        self.channels = [
            int(64 * width_mult)
            for width_mult in self.FLAGS['width_mult_list']
        ]

        self.stem = nn.Sequential(
            SlimmableConv2d(
                [3 for _ in range(len(self.channels))],
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            SwitchableBatchNorm2d(self.channels),
        )

        self._make_block()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = SlimmableLinear(
            self.candidate_Cs, [self._num_classes], bias=False)

    def forward(self, x, forward_op=None):
        """
        forward_op: [4, 0, 2, 4, 6]
        """
        assert forward_op is not None
        # outs = []
        # stem
        idx = forward_op[0]
        x = self.stem[0](x, idx, idx)
        x = self.stem[1](x, idx)
        # blocks
        for i, block in enumerate(self._blocks):
            pre_op = forward_op[sum(self._op_layers_list[:i]) -
                                1] if i > 0 else -1
            x = block(
                x,
                i,
                forward_list=forward_op[sum(self._op_layers_list[:i]
                                            ):sum(self._op_layers_list[:(i +
                                                                         1)])],
                pre_op=pre_op,
            )

        x = self.gap(x)
        x = x.view(-1, self.candidate_Cs[forward_op[-1]])

        return self.classifier(x, forward_op[-1], 0)

    def set_forward_cfg(self, method='fair'):  # support method: uniform/fair
        # TODO: support fair
        if method == 'uni':
            forward_op = uniform_random_op_encoding(
                num_of_ops=self._num_of_ops, layers=sum(self._op_layers_list))
        elif method == 'fair':
            forward_op = fair_random_op_encoding(
                num_of_ops=self._num_of_ops, layers=sum(self._op_layers_list))
        elif method == 'large':
            forward_op = [7 for _ in range(sum(self._op_layers_list))]
        elif method == 'small':
            forward_op = [0 for _ in range(sum(self._op_layers_list))]
        else:
            raise NotImplementedError
        return forward_op

    def init_weights(self):
        pass  # FIXME: Using pytorch default weight init

    def _make_block(self, ):
        self._blocks = nn.ModuleList()
        for i, layers in enumerate(self._op_layers_list):
            self._blocks.append(Block(layers, stage=i, target=self.target))

    def get_all_path(self, start_block=0):
        return all_op_encoding(
            num_of_ops=self._num_of_ops,
            layers=self._op_layers_list[start_block])

    def reset_params(self):
        self.apply(reset)

    def step_start_trigger(self):
        """generate fair choices"""
        pass

    def get_layers(self, block):
        """get num layers of a block"""
        return self.block_cfgs[block][3]

    def get_block(self, block_num):
        """get block module to train separately"""
        return self._blocks[block_num]


def get_model_parameters_number(model):
    return sum(p.numel() for p in model.parameters())  # if p.requires_grad)
