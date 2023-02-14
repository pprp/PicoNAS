from typing import List

import torch.nn as nn
from torch.nn import Sequential

from piconas.nas.mutables import OneShotOP
from piconas.utils.misc import convert_arch2dict
from piconas.models.registry import register_model


class ConvBNReLU(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, kernel_size, stride, expand_ratio,
                 use_res_connect):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))

        self.kernel_size = kernel_size
        self.use_res_connect = use_res_connect

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
            layers.extend([
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    kernel_size=self.kernel_size,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class Identity(nn.Module):
    """
    Identity cell
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


@register_model
class MacroBenchmarkSuperNet(nn.Module):
    """MacroBenchmark SuperNet.
    There are 14 searchable cells and 2 choices in each cell.
    Downsampling cells are replaced at 4,9,14 layer.

    Note:
        Each cell have three choice types, which are `I`, `1`, and `2`:
        1. Identity: a placeholder to allow nets of different depths
            (encoded as 'I')
        2. MBConv with expansion rate 3, kernel size 3x3 (encoded as '1')
        3. MBConv with expansion rate 6, kernel size 5x5 (encoded as '2')

    Args:
        num_classes (int, optional): _description_. Defaults to 10.
        first_conv_out_channels (int, optional): _description_. Defaults to 32.
    """

    def __init__(self, num_classes: int = 10, first_conv_out_channels=32):

        super(MacroBenchmarkSuperNet, self).__init__()

        self.arch_settings: List[List] = [
            # channel, num_blocks, stride, type
            [32, 4, 1, 'mutable'],  # 0-3
            [32, 1, 2, 'downsample'],  # 4
            [64, 4, 1, 'mutable'],  # 5-8
            [64, 1, 2, 'downsample'],  # 9
            [128, 4, 1, 'mutable'],  # 10-13
            [128, 1, 2, 'downsample'],  # 14
            [256, 2, 1, 'mutable'],  # 15, 16
        ]
        self.in_conv = ConvBNReLU(
            3, first_conv_out_channels, kernel_size=3, stride=1)

        self.in_channels = first_conv_out_channels

        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            channel, num_blocks, stride, block_type = layer_cfg
            inverted_res_layer = self._make_layer(
                out_channels=channel,
                num_blocks=num_blocks,
                stride=stride,
                block_type=block_type,
            )
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        self.features_mixing = ConvBNReLU(256, 1280, kernel_size=1, stride=1)

        self.out1 = nn.AdaptiveAvgPool2d((1, 1))
        self.out2 = nn.Linear(1280, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int,
                    block_type: str) -> Sequential:
        """make single layer"""
        if block_type == 'downsample':
            layers = []
            layers += [
                InvertedResidual(
                    out_channels,
                    out_channels * 2,
                    kernel_size=3,
                    stride=2,
                    expand_ratio=3,
                    use_res_connect=0,
                )
            ]
            return Sequential(*layers)

        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1

            candidate_ops = nn.ModuleDict({
                'I':
                Identity(),
                '1':
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    expand_ratio=3,
                    use_res_connect=1,
                ),
                '2':
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=stride,
                    expand_ratio=6,
                    use_res_connect=1,
                ),
            })
            osop = OneShotOP(candidate_ops=candidate_ops)
            layers.append(osop)

            self.in_channels = out_channels

        self.in_channels *= 2

        return Sequential(*layers)

    def forward(self, x):

        # stem convolution
        x = self.in_conv(x)

        # main body (list of cells)
        for layer_name in self.layers:
            layer = getattr(self, layer_name)
            x = layer(x)

        # classifier
        x = self.features_mixing(x)
        x = self.out1(x)
        x = x.view(x.shape[0], -1)
        x = self.out2(x)
        return x

    def forward_distill(self, x):
        # for compute flops and params
        x = self.in_conv(x)

        # main body (list of cells)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)

        # classifier
        x = self.features_mixing(x)
        x = self.out1(x)
        feat = x.view(x.shape[0], -1)
        x = self.out2(feat)
        return x, feat


if __name__ == '__main__':
    arch_config = 'I1I1221I121121'
    from piconas.nas.mutators import OneShotMutator

    supernet = MacroBenchmarkSuperNet()

    mutator = OneShotMutator()
    mutator.prepare_from_supernet(supernet)

    sg = mutator.search_group

    print(sg.keys())

    print(mutator.random_subnet)

    mutator.set_subnet(convert_arch2dict(arch_config))
