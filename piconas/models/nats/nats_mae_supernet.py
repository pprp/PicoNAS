"""
License: https://github.com/changlin31/BossNAS
"""

from typing import List

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from piconas.models.nats.nats_ops import SlimmableConv2d
from ..registry import register_model
from .nats_supernet import SupernetNATS


@register_model
class MAESupernetNATS(SupernetNATS):
    def __init__(self, target='cifar10') -> None:
        super().__init__(target=target)

        self.last_channel = 128

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # convert from dynamic to static
        self.last_dynamic_conv = SlimmableConv2d(
            self.candidate_Cs,
            [self.last_channel for _ in range(len(self.candidate_Cs))],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.last_bn = nn.BatchNorm2d(self.last_channel)

        # decoder is two upsample layers
        self.decoder = nn.Sequential(
            # x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                self.last_channel,
                self.last_channel // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(self.last_channel // 2),
            nn.ReLU(inplace=True),
            # x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                self.last_channel // 2, 3, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def process_mask(self, x: Tensor, mask: Tensor, num_patch=16):
        # process masked image
        x = rearrange(
            x, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=num_patch, p2=num_patch
        )
        mask = rearrange(mask, 'b h w -> b (h w)')
        mask = mask.unsqueeze(-1).repeat(1, 1, 12)
        x = x * mask
        x = rearrange(
            x,
            'b (p1 p2) (c h w) -> b c (p1 h) (p2 w)',
            p1=num_patch,
            p2=num_patch,
            c=3,
            h=2,
            w=2,
        )
        return x

    def forward(self, x: Tensor, mask: Tensor, forward_op: List = None) -> Tensor:
        # process mask
        x = self.process_mask(x, mask)

        # forward the masked image
        assert forward_op is not None
        # stem
        idx = forward_op[0]
        x = self.stem[0](x, idx, idx)
        x = self.stem[1](x, idx)
        # blocks
        for i, block in enumerate(self._blocks):
            pre_op = forward_op[sum(self._op_layers_list[:i]) - 1] if i > 0 else -1
            x = block(
                x,
                i,
                forward_list=forward_op[
                    sum(self._op_layers_list[:i]) : sum(self._op_layers_list[: (i + 1)])
                ],
                pre_op=pre_op,
            )

        # convert from dynamic to static
        x = self.last_dynamic_conv(x, forward_op[-1], 0)
        x = self.last_bn(x)

        feat = self.avgpool(x)
        feat = feat.view(feat.size(0), -1)

        return self.decoder(x), feat
