import torch.nn as nn
from einops import rearrange
from torch import Tensor

from pplib.models.spos.spos_modules import ShuffleModule, ShuffleXModule
from pplib.nas.mutables import OneShotOP
from ..registry import register_model


@register_model
class SearchableShuffleNetV2(nn.Module):

    def __init__(self, classes=10) -> None:
        super().__init__()

        self.arch_settings = [
            # channel, num_blocks, stride
            [64, 4, 1],
            [160, 4, 2],
            [320, 8, 2],
            [640, 4, 1],
        ]
        self.in_channels = 16
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

        self.layers = nn.ModuleList()
        for channel, num_blocks, stride in self.arch_settings:
            layer = self._make_layer(channel, num_blocks, stride)
            self.layers.append(layer)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel, affine=False),
            nn.ReLU6(inplace=True))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.last_channel, classes, bias=False)

    def _make_layer(self, out_channels: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            if i == 0 and stride == 2:
                inp, outp, stride = self.in_channels, out_channels, 2
            else:
                inp, outp, stride = self.in_channels // 2, out_channels, 1
            stride = 2 if stride == 2 and i == 0 else 1
            candidate_ops = nn.ModuleDict({
                'shuffle_3x3':
                ShuffleModule(inp, outp, kernel=3, stride=stride),
                'shuffle_5x5':
                ShuffleModule(inp, outp, kernel=5, stride=stride),
                'shuffle_7x7':
                ShuffleModule(inp, outp, kernel=7, stride=stride),
                'shuffle_xception':
                ShuffleXModule(inp, outp, stride=stride),
            })
            layers.append(OneShotOP(candidate_ops=candidate_ops))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.first_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.last_conv(x)
        x = self.gap(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


@register_model
class SearchableMAE(SearchableShuffleNetV2):

    def __init__(self, classes=10) -> None:
        super().__init__(classes=classes)

        self.decoder = nn.Sequential(
            # x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                self.last_channel,
                self.last_channel // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            nn.BatchNorm2d(self.last_channel // 2),
            nn.ReLU(inplace=True),

            # x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                self.last_channel // 2,
                3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def process_mask(self, x: Tensor, mask: Tensor, patch_size=16):
        # process masked image
        x = rearrange(
            x,
            'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)',
            p1=patch_size,
            p2=patch_size)
        mask = rearrange(mask, 'b h w -> b (h w)')
        mask = mask.unsqueeze(-1).repeat(1, 1, 12)
        x = x * mask
        x = rearrange(
            x,
            'b (p1 p2) (c h w) -> b c (p1 h) (p2 w)',
            p1=patch_size,
            p2=patch_size,
            c=3,
            h=2,
            w=2)
        return x

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:  # type: ignore
        # process mask
        x = self.process_mask(x, mask)

        # forward the masked img
        x = self.first_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.last_conv(x)
        return self.decoder(x)


if __name__ == '__main__':
    m = SearchableShuffleNetV2()
    import torch

    inputs = torch.randn(4, 3, 32, 32)

    outputs = m(inputs)
    print(outputs.shape)

    m = SearchableMAE()
    outputs = m(inputs)
    print(outputs.shape)
