import torch
import torch.nn as nn

from piconas.nas.mutables import DiffOP
from ..registry import register_model


class ReLUConvBN(nn.Module):
    """
    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    padding: int
        zero-padding added to both sides of the input
    dilation: int
        spacing between kernel elements
    bn_affine: bool
        If set to ``True``, ``torch.nn.BatchNorm2d`` will have
        learnable affine parameters. Default: True
    bn_momentun: float
        the value used for the running_mean and running_var
        computation. Default: 0.1
    bn_track_running_stats: bool
        When set to ``True``, ``torch.nn.BatchNorm2d`` tracks the
        running mean and variance. Default: True
    """

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        bn_affine=False,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                C_out,
                affine=bn_affine,
                momentum=bn_momentum,
                track_running_stats=bn_track_running_stats,
            ),
        )

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        return self.op(x)


class Pooling(nn.Module):
    """
    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    bn_affine: bool
        If set to ``True``, ``torch.nn.BatchNorm2d`` will
        have learnable affine parameters. Default: True
    bn_momentun: float
        the value used for the running_mean and running_var
        computation. Default: 0.1
    bn_track_running_stats: bool
        When set to ``True``, ``torch.nn.BatchNorm2d`` tracks
        the running mean and variance. Default: True
    """

    def __init__(
        self,
        C_in,
        C_out,
        stride,
        bn_affine=False,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(Pooling, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 0, bn_affine, bn_momentum, bn_track_running_stats
            )
        self.op = nn.AvgPool2d(
            3, stride=stride, padding=1, count_include_pad=False)

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        if self.preprocess:
            x = self.preprocess(x)
        return self.op(x)


class Zero(nn.Module):
    """
    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    """

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            return x.new_zeros(shape, dtype=x.dtype, device=x.device)


class FactorizedReduce(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        bn_affine=False,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(C_in, C_outs[i], 1,
                              stride=stride, padding=0, bias=False)
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(
            C_out,
            affine=bn_affine,
            momentum=bn_momentum,
            track_running_stats=bn_track_running_stats,
        )

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat(
            [self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class NASBench201Cell(nn.Module):
    """
    Builtin cell structure of NAS Bench 201. One cell contains
    four nodes. The First node serves as an input node
    accepting the output of the previous cell. And other nodes
    connect to all previous nodes with an edge that represents
    an operation chosen from a set to transform the tensor from
    the source node to the target node. Every node accepts all
    its inputs and adds them as its output.

    Args:
        cell_id: str
            the name of this cell
        C_in: int
            the number of input channels of the cell
        C_out: int
            the number of output channels of the cell
        stride: int
            stride of all convolution operations in the cell
        bn_affine: bool
            If set to ``True``, all ``torch.nn.BatchNorm2d`` in
            this cell will have learnable affine parameters. Default: True
        bn_momentum: float
            the value used for the running_mean and running_var
            computation. Default: 0.1
        bn_track_running_stats: bool
            When set to ``True``, all ``torch.nn.BatchNorm2d``
            in this cell tracks the running mean and variance. Default: True
    """

    def __init__(
        self,
        cell_id,
        C_in,
        C_out,
        stride,
        bn_affine=False,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(NASBench201Cell, self).__init__()

        self.NUM_NODES = 4
        self.layers = nn.ModuleList()

        for i in range(self.NUM_NODES):
            node_ops = nn.ModuleList()
            for layer_idx in range(i):
                candidate_op = nn.ModuleDict(
                    {
                        'none': Zero(C_in, C_out, stride),
                        'avg_pool_3x3': Pooling(
                            C_in,
                            C_out,
                            stride if layer_idx == 0 else 1,
                            bn_affine,
                            bn_momentum,
                            bn_track_running_stats,
                        ),
                        'nor_conv_3x3': ReLUConvBN(
                            C_in,
                            C_out,
                            3,
                            stride if layer_idx == 0 else 1,
                            1,
                            1,
                            bn_affine,
                            bn_momentum,
                            bn_track_running_stats,
                        ),
                        'nor_conv_1x1': ReLUConvBN(
                            C_in,
                            C_out,
                            1,
                            stride if layer_idx == 0 else 1,
                            0,
                            1,
                            bn_affine,
                            bn_momentum,
                            bn_track_running_stats,
                        ),
                        'skip_connect': nn.Identity()
                        if stride == 1 and C_in == C_out
                        else FactorizedReduce(
                            C_in,
                            C_out,
                            stride if layer_idx == 0 else 1,
                            bn_affine,
                            bn_momentum,
                            bn_track_running_stats,
                        ),
                    }
                )
                node_ops.append(
                    DiffOP(candidate_ops=candidate_op,
                           alias=f'node{i}_edge{layer_idx}')
                )
            self.layers.append(node_ops)
        self.in_dim = C_in
        self.out_dim = C_out
        self.cell_id = cell_id

    def forward(self, input):
        nodes = [input]
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]


class ResNetBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride,
        bn_affine=False,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(ResNetBasicBlock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(
            inplanes,
            planes,
            3,
            stride,
            1,
            1,
            bn_affine,
            bn_momentum,
            bn_track_running_stats,
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, bn_affine, bn_momentum, bn_track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes,
                planes,
                1,
                1,
                0,
                1,
                bn_affine,
                bn_momentum,
                bn_track_running_stats,
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)
        return inputs + basicblock


@register_model
class DiffNASBench201Network(nn.Module):
    def __init__(
        self,
        stem_out_channels=16,
        num_modules_per_stack=5,
        bn_affine=False,
        bn_momentum=0.1,
        bn_track_running_stats=True,
        num_classes=10,
    ):
        super(DiffNASBench201Network, self).__init__()
        self.channels = C = stem_out_channels
        self.num_modules = N = num_modules_per_stack
        self.num_labels = num_classes

        self.bn_momentum = bn_momentum
        self.bn_affine = bn_affine
        self.bn_track_running_stats = bn_track_running_stats

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=self.bn_momentum),
        )

        layer_channels = [C] * N + [C * 2] + \
            [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + \
            [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicBlock(
                    C_prev,
                    C_curr,
                    2,
                    self.bn_affine,
                    self.bn_momentum,
                    self.bn_track_running_stats,
                )
            else:
                cell = NASBench201Cell(
                    i,
                    C_prev,
                    C_curr,
                    1,
                    self.bn_affine,
                    self.bn_momentum,
                    self.bn_track_running_stats,
                )
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, momentum=self.bn_momentum), nn.ReLU(
                inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits
