import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from piconas.models.registry import register_model
from piconas.nas.mutables import OneShotOP


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
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 0, bn_affine,
                                         bn_momentum, bn_track_running_stats)
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
            return (x.mul(0.0) if self.stride == 1 else
                    x[:, :, ::self.stride, ::self.stride].mul(0.0))

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
        if stride != 2:
            raise ValueError('Invalid stride : {:}'.format(stride))
        C_outs = [C_out // 2, C_out - C_out // 2]
        self.convs = nn.ModuleList()
        for i in range(2):
            self.convs.append(
                nn.Conv2d(
                    C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.bn = nn.BatchNorm2d(
            C_out,
            affine=bn_affine,
            momentum=bn_momentum,
            track_running_stats=bn_track_running_stats,
        )

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])],
                        dim=1)
        out = self.bn(out)
        return out


class NASBench201Cell(nn.Module):
    """
    Builtin cell structure of NAS Bench 201. One cell contains
    four nodes. The First node serves as an input node
    accepting the output of the previous cell. And other nodes
    connect to all previous nodes with an edge that
    represents an operation chosen from a set to transform the
    tensor from the source node to the target node.
    Every node accepts all its inputs and adds them as its output.

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
        cell_id: int,
        C_in: int,
        C_out: int,
        stride: int,
        bn_affine=False,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(NASBench201Cell, self).__init__()

        self.NUM_NODES = 4
        self.layers = nn.ModuleList()
        self.in_dim = C_in
        self.out_dim = C_out
        self.cell_id = cell_id

        for i in range(self.NUM_NODES):
            node_ops = nn.ModuleList()
            for layer_idx in range(i):
                candidate_op = nn.ModuleDict({
                    # 'none':
                    # Zero(C_in, C_out, stride),
                    'avg_pool_3x3':
                    Pooling(
                        C_in,
                        C_out,
                        stride if layer_idx == 0 else 1,
                        bn_affine,
                        bn_momentum,
                        bn_track_running_stats,
                    ),
                    'nor_conv_3x3':
                    ReLUConvBN(
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
                    'nor_conv_1x1':
                    ReLUConvBN(
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
                    'skip_connect':
                    nn.Identity()
                    if stride == 1 and C_in == C_out else FactorizedReduce(
                        C_in,
                        C_out,
                        stride if layer_idx == 0 else 1,
                        bn_affine,
                        bn_momentum,
                        bn_track_running_stats,
                    ),
                })
                node_ops.append(
                    OneShotOP(
                        candidate_ops=candidate_op,
                        alias=f'node{i}_edge{layer_idx}'))  # with alias
            self.layers.append(node_ops)

    def forward(self, input):
        nodes = [input]
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]


class ResNetBasicBlock(nn.Module):
    """Basic block of resnet

    Args:
        inplanes (_type_): _description_
        planes (_type_): _description_
        stride (_type_): _description_
        bn_affine (bool, optional): _description_. Defaults to False.
        bn_momentum (float, optional): _description_. Defaults to 0.1.
        bn_track_running_stats (bool, optional): _description_. Defaults to True.
        with_residual (bool, optional): When used by ZenNAS, set to False.
            Defaults to True.
    """

    def __init__(
        self,
        inplanes,
        planes,
        stride,
        bn_affine=False,
        bn_momentum=0.1,
        bn_track_running_stats=True,
        with_residual=True,
    ):

        super(ResNetBasicBlock, self).__init__()
        assert stride in [1, 2], 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            bn_affine=bn_affine,
            bn_momentum=bn_momentum,
            bn_track_running_stats=bn_track_running_stats,
        )
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, bn_affine,
                                 bn_momentum, bn_track_running_stats)
        if stride == 2:
            # downsample the input if the input and output shapes are different
            if inplanes != planes:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False),
                )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1,
                                         bn_affine, bn_momentum,
                                         bn_track_running_stats)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2
        self.with_residual = with_residual

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)

        return inputs + basicblock if self.with_residual else basicblock


@register_model
class MIMOSNASBench201Network(nn.Module):

    def __init__(
        self,
        stem_out_channels: int = 16,
        num_modules_per_stack: int = 5,
        bn_affine: bool = False,
        bn_momentum: float = 0.1,
        bn_track_running_stats: bool = True,
        num_classes: int = 10,
        with_residual: bool = True,
    ):
        super(MIMOSNASBench201Network, self).__init__()
        self.channels = C = stem_out_channels
        self.num_modules = N = num_modules_per_stack
        self.num_classes = num_classes
        self.with_residual = with_residual

        self.bn_momentum = bn_momentum
        self.bn_affine = bn_affine
        self.bn_track_running_stats = bn_track_running_stats

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=self.bn_momentum),
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4
                                                            ] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [
            True
        ] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for i, (C_curr,
                reduction) in enumerate(zip(layer_channels, layer_reductions)):

            if reduction:
                cell = ResNetBasicBlock(C_prev, C_curr, 2, self.bn_affine,
                                        self.bn_momentum,
                                        self.bn_track_running_stats,
                                        self.with_residual)
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

        # decoder is two upsample layers
        self.decoder = nn.Sequential(
            # x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                C_prev,
                C_prev // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(C_prev // 2),
            nn.ReLU(inplace=True),
            # x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                C_prev // 2, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, momentum=self.bn_momentum),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_classes)

    def process_mask(self, x: Tensor, mask: Tensor, num_patch=16):
        # rearrange x into (batch_size, num_patches, num_channels * patch_height * patch_width)
        x = rearrange(
            x,
            'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)',
            p1=num_patch,
            p2=num_patch)
        # rearrange mask into (batch_size, num_patches)
        mask = rearrange(mask, 'b h w -> b (h w)')
        # repeat mask to match x channel size
        mask = mask.unsqueeze(-1).repeat(1, 1, 12)
        # multiply masked image by mask
        x = x * mask
        # rearrange x into (batch_size, num_channels, num_patches * patch_height * patch_width)
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

    def forward(self, inputs: Tensor, mask: Tensor = None):
        """
        Args:
            inputs: input images
            mask: mask for masked image

        Returns:
            decoder output, classifier output
        """
        if mask is not None:
            inputs = self.process_mask(inputs, mask)

        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        o1 = self.decoder(out)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)

        o2 = self.classifier(out)
        return o1, o2