import torch
import torch.nn as nn

from pplib.models.bnnas.bn_blocks import InvertedResidual, blocks_dict, conv_bn
from pplib.nas.mutables.oneshot_mutable import OneShotOP
from ..registry import register_model


@register_model
class BNNAS(nn.Module):

    def __init__(
        self,
        first_stride: int = 1,
        first_channels: int = 24,
        width_mult: int = 1,
        num_classes: int = 1000,
        is_only_train_bn: bool = True,
    ):
        super(BNNAS, self).__init__()

        self.arch_settings = [
            # channel, num_blocks, stride
            [32, 2, 1],  # origin 2
            [40, 2, 2],
            [80, 4, 2],
            [96, 4, 1],
            [192, 4, 2],
            [320, 1, 1],
        ]

        self.in_channels = first_channels
        self.last_channels = 320

        self.in_channels = int(self.in_channels * width_mult)

        self.first_conv = nn.Sequential(
            conv_bn(3, 40, first_stride),
            InvertedResidual(40, self.in_channels, 3, 1, 1, 1),
        )

        self.layers = nn.ModuleList()
        for channel, num_blocks, stride in self.arch_settings:
            layer = self._make_layer(channel, num_blocks, stride)
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.last_channels, num_classes))

        if is_only_train_bn:
            print('Only train BN.')
            self.freeze_except_bn()

    def _make_layer(self, out_channels, num_blocks, stride) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            if i == 0 and stride == 2:
                inp, outp, stride = self.in_channels, out_channels, 2
            else:
                inp, outp, stride = self.in_channels, out_channels, 1
            stride = 2 if stride == 2 and i == 0 else 1
            candidate_ops = nn.ModuleDict({
                'bn_k3r3':
                blocks_dict['k3r3'](inp, outp, stride),
                'bn_k3r6':
                blocks_dict['k3r6'](inp, outp, stride),
                'bn_k5r3':
                blocks_dict['k5r3'](inp, outp, stride),
                'bn_k5r6':
                blocks_dict['k5r6'](inp, outp, stride),
                'bn_k7r3':
                blocks_dict['k7r3'](inp, outp, stride),
                'bn_k7r6':
                blocks_dict['k7r6'](inp, outp, stride),
            })
            layers.append(OneShotOP(candidate_ops=candidate_ops))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_conv(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channels)
        x = self.classifier(x)
        return x

    def freeze_except_bn(self):
        for _, params in self.named_parameters():
            params.requires_grad = False

        for _, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.requires_grad_(True)

    def defrost_all_params(self):
        for _, params in self.named_parameters():
            params.requires_grad = True

    def freeze_all_params(self):
        for _, params in self.named_parameters():
            params.requires_grad = False


if __name__ == '__main__':
    m = BNNAS(1)
    input = torch.randn(8, 3, 32, 32)
    output = m(input)
    print(output.shape)
