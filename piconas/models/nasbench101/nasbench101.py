import math
import random

import numpy as np
import torch
import torch.nn as nn

from piconas.models.nasbench101.nb101_blocks import ConvBnRelu, MaxPool
from piconas.models.registry import register_model
from piconas.nas.mutables import OneShotPathOP


class Cell(nn.Module):

    def __init__(self, inplanes, outplanes, shadow_bn, layer_idx=0):
        super(Cell, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.shadow_bn = shadow_bn

        nodes = nn.ModuleList([])
        # 12 + 1

        for i in range(4):
            nodes.append(ConvBnRelu(self.inplanes, self.outplanes, 1))
            nodes.append(ConvBnRelu(self.inplanes, self.outplanes, 3))
            nodes.append(MaxPool(self.inplanes, self.outplanes))
        nodes.append(nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1))
        self.edges = OneShotPathOP(
            candidate_ops=nodes, alias=f'layer-{layer_idx}')

        self.bn_list = nn.ModuleList([])
        if self.shadow_bn:
            for j in range(4):
                self.bn_list.append(nn.BatchNorm2d(outplanes))
        else:
            self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        return self.edges(x)


@register_model
class OneShotNASBench101Network(nn.Module):

    def __init__(self, init_channels=128, num_classes=10, shadow_bn=True):
        super(OneShotNASBench101Network, self).__init__()
        self.init_channels = init_channels

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                self.init_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(inplace=True),
        )

        self.cell_list = nn.ModuleList([])
        for i in range(9):
            if i in [3, 6]:
                # downsample
                self.cell_list.append(
                    Cell(
                        self.init_channels,
                        self.init_channels * 2,
                        shadow_bn=shadow_bn,
                        layer_idx=i))
                self.init_channels *= 2
            else:
                self.cell_list.append(
                    Cell(
                        self.init_channels,
                        self.init_channels,
                        shadow_bn=shadow_bn,
                        layer_idx=i))

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.init_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        for i in range(9):
            x = self.cell_list[i](x)
            if i in [2, 5]:
                x = nn.MaxPool2d(2, 2, padding=0)(x)
        x = self.global_pooling(x)
        x = x.view(-1, self.init_channels)
        out = self.classifier(x)
        return out


def random_choice(m):
    assert m >= 1

    choice = {}
    m_ = np.random.randint(low=1, high=m + 1, size=1)[0]
    path_list = random.sample(range(m), m_)

    ops = []
    for i in range(m_):
        ops.append(random.sample(range(3), 1)[0])
        # ops.append(random.sample(range(2), 1)[0])

    choice['op'] = ops
    choice['path'] = path_list

    return choice


if __name__ == '__main__':
    # ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    # choice = {'path': [0, 1, 2],  # a list of shape (4, )
    #           'op': [0, 0, 0]}  # possible shapes: (), (1, ), (2, ), (3, )

    # op: [2, 0]  path: [0, 2]
    from piconas.nas.mutators import OneShotMutator

    choice = random_choice(3)
    print(choice)

    model = OneShotNASBench101Network(init_channels=128)
    mutator = OneShotMutator(with_alias=True)
    mutator.prepare_from_supernet(model)

    rand_subnet = mutator.random_subnet
    print(rand_subnet)
    mutator.set_subnet(rand_subnet)

    input = torch.randn((1, 3, 32, 32))
    print(model.forward(input).shape)
