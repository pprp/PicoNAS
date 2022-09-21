"""
Code below from NASBench-201 and slighly adapted
"""

import torch.nn as nn

from ..core.primitives import AbstractPrimitive, ReLUConvBN


class ResNetBasicblock(AbstractPrimitive):

    def __init__(self, C_in, C_out, stride, affine=False):
        super().__init__(locals())
        assert stride in [1, 2], 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(C_in, C_out, 3, stride)
        self.conv_b = ReLUConvBN(C_out, C_out, 3)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    C_in,
                    C_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x, edge_data):
        basicblock = self.conv_a(x, None)
        basicblock = self.conv_b(basicblock, None)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock

    forward_beforeGP = forward

    def get_embedded_ops(self):
        return None
