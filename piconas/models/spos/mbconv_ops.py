import torch.nn as nn
from timm.models.efficientnet_blocks import (
    SqueezeExcite,
    make_divisible,
    resolve_se_args,
)
from timm.models.layers import create_conv2d, drop_path

BN_ARGS = dict(momentum=0.1, eps=1e-5)

OPS = {
    # MBConv
    'MB6_3x3_se0.25': lambda in_channels, out_channels, stride, downsample: InvertedResidual(
        in_channels,
        out_channels,
        3,
        stride,
        act_layer=nn.SiLU,
        downsample=downsample,
        exp_ratio=6.0,
        se_ratio=0.25,
        norm_kwargs=BN_ARGS,
    ),
    'MB6_5x5_se0.25': lambda in_channels, out_channels, stride, downsample: InvertedResidual(
        in_channels,
        out_channels,
        5,
        stride,
        act_layer=nn.SiLU,
        downsample=downsample,
        exp_ratio=6.0,
        se_ratio=0.25,
        norm_kwargs=BN_ARGS,
    ),
    'MB3_3x3_se0.25': lambda in_channels, out_channels, stride, downsample: InvertedResidual(
        in_channels,
        out_channels,
        3,
        stride,
        act_layer=nn.SiLU,
        downsample=downsample,
        exp_ratio=3.0,
        se_ratio=0.25,
        norm_kwargs=BN_ARGS,
    ),
    'MB3_5x5_se0.25': lambda in_channels, out_channels, stride, downsample: InvertedResidual(
        in_channels,
        out_channels,
        5,
        stride,
        act_layer=nn.SiLU,
        downsample=downsample,
        exp_ratio=3.0,
        se_ratio=0.25,
        norm_kwargs=BN_ARGS,
    ),
}


class InvertedResidual(nn.Module):
    """
    modified from timm.models.efficientnet_blocks.InvertedResidual
    Inverted residual block w/ optional SE and CondConv routing
    w/ optional down-sample residual connection
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        dilation=1,
        pad_type='',
        act_layer=nn.ReLU,
        noskip=False,
        exp_ratio=1.0,
        exp_kernel_size=1,
        pw_kernel_size=1,
        se_ratio=0.0,
        se_kwargs=None,
        norm_layer=nn.BatchNorm2d,
        norm_kwargs=None,
        conv_kwargs=None,
        drop_path_rate=0.0,
        downsample=None,
    ):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = not noskip
        self.downsample = downsample
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(
            in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs
        )
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=stride,
            dilation=dilation,
            padding=pad_type,
            depthwise=True,
            **conv_kwargs
        )
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs
        )
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        # sourcery skip: assign-if-exp, inline-immediately-returned-variable
        if location == 'expansion':  # after SE, input to PWL
            return dict(
                module='conv_pwl',
                hook_type='forward_pre',
                num_chs=self.conv_pwl.in_channels,
            )
        else:  # location == 'bottleneck', block output
            return dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual

        return x
