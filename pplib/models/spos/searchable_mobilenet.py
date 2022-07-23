import torch.nn as nn
from torch import Tensor

from pplib.models.spos.spos_modules import InvertedResidual, blocks_dict
from pplib.nas.mutables import OneShotOP


class SearchableMobileNet(nn.Module):

    def __init__(self, classes: int = 10, width_mult: float = 1.) -> None:
        super().__init__()
        self.width_mult = width_mult
        self.arch_settings = [
            # channel, num_blocks, stride
            [32, 4, 1],  # 2 for imagenet
            [56, 4, 2],
            [112, 4, 2],
            [128, 4, 1],
            [256, 4, 1],  # 2 for imagenet
            [432, 1, 1],
        ]
        self.in_channels = int(40 * width_mult)
        self.last_channel = 640

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                3,
                self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False), nn.BatchNorm2d(self.in_channels, affine=False),
            nn.ReLU6(inplace=True))

        self.stem_MBConv = InvertedResidual(self.in_channels,
                                            int(24 * width_mult), 3, 1, 1, 1)

        self.in_channels = int(24 * width_mult)

        self.layers = nn.ModuleList()
        for channel, num_blocks, stride in self.arch_settings:
            layer = self._make_layer(channel, num_blocks, stride)
            self.layers.append(layer)

        # building last several layers
        self.last_conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel, affine=False),
            nn.ReLU6(inplace=True))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.last_channel, classes, bias=False)

    def _make_layer(self, out_channels: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        layers = []
        out_channels = int(out_channels * self.width_mult)

        for i in range(num_blocks):
            if i == 0:
                inp, outp, stride = self.in_channels, out_channels, stride
            else:
                inp, outp, stride = self.in_channels, out_channels, 1

            candidate_ops = nn.ModuleDict({
                'mbconv_k3_r3':
                blocks_dict['mobilenet_3x3_ratio_3'](inp, outp, stride),
                'mbconv_k3_r6':
                blocks_dict['mobilenet_3x3_ratio_6'](inp, outp, stride),
                'mbconv_k5_r3':
                blocks_dict['mobilenet_5x5_ratio_3'](inp, outp, stride),
                'mbconv_k5_r6':
                blocks_dict['mobilenet_5x5_ratio_6'](inp, outp, stride),
                'mbconv_k7_r3':
                blocks_dict['mobilenet_7x7_ratio_3'](inp, outp, stride),
                'mbconv_k7_r6':
                blocks_dict['mobilenet_7x7_ratio_6'](inp, outp, stride),
            })
            layers.append(OneShotOP(candidate_ops=candidate_ops))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.first_conv(x)
        x = self.stem_MBConv(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.last_conv(x)
        x = self.gap(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = SearchableMobileNet()
