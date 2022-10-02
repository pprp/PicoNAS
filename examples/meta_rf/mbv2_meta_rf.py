# https://github.com/Andrew-Qibin/CoordAttention/blob/main/mbv2_ca.py

import math

import torch
import torch.nn as nn

from pplib.models.registry import register_model
from pplib.nas.mutables import DynaDiffOP


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MetaReceptiveField_v1(nn.Module):
    """Adjust Receptive Field by Meta Learning."""

    def __init__(self, in_channels=128):
        super().__init__()
        candidate_ops = nn.ModuleDict({
            'skip_connect':
            Identity(),
            'coord_att':
            CoordAtt(in_channels, in_channels),
            'max_pool_3x3':
            nn.MaxPool2d(3, stride=1, padding=1),
            'max_pool_5x5':
            nn.MaxPool2d(5, stride=1, padding=2),
            'max_pool_7x7':
            nn.MaxPool2d(7, stride=1, padding=3),
        })
        self.meta_rf = DynaDiffOP(candidate_ops, dyna_thresh=0.3)

    def forward(self, x):
        return self.meta_rf(x)


class SplitBlock(nn.Module):

    def __init__(self, ratio=0.5):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class MetaReceptiveField_v2(nn.Module):
    """Adjust Receptive Field by Meta Learning with split block."""

    def __init__(self, in_channels=128, ratio: float = 0.5):
        super().__init__()
        in_channels = int(in_channels * ratio)

        candidate_ops = nn.ModuleDict({
            'skip_connect':
            Identity(),
            'coord_att':
            CoordAtt(in_channels, in_channels),
            'max_pool_3x3':
            nn.MaxPool2d(3, stride=1, padding=1),
            'max_pool_5x5':
            nn.MaxPool2d(5, stride=1, padding=2),
            'max_pool_7x7':
            nn.MaxPool2d(7, stride=1, padding=3),
        })
        self.meta_rf = DynaDiffOP(candidate_ops)
        self.split = SplitBlock(ratio)

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.meta_rf(x2)
        return torch.cat([x1, out], 1)


MetaReceptiveField = MetaReceptiveField_v2


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
                bias=False), nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # meta receptive field
                MetaReceptiveField(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


@register_model
class MobileNetv2MetaReceptionField(nn.Module):
    """MobileNetV2 + Meta RF

    Args:
        num_classes (int, optional): _description_. Defaults to 1000.
        width_mult (_type_, optional): _description_. Defaults to 1..
    """

    def __init__(self, num_classes=1000, width_mult=1.):
        super().__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult,
                                        4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult,
                                             4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(
                    block(input_channel, output_channel, s if i == 0 else 1,
                          t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(
            1280 * width_mult,
            4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(output_channel, num_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    m = MetaReceptiveField_v2(in_channels=6)
    i = torch.randn(3, 6, 32, 32)
    o = m(i)
    print(o.shape)
