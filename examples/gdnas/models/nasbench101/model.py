"""Builds the Pytorch computational graph.
Tensors flowing into a single vertex are added together for all vertices
except the output, which is concatenated instead. Tensors flowing out of input
are always added.
If interior edge channels don't match, drop the extra channels (channels are
guaranteed non-decreasing). Tensors flowing out of the input as always
projected instead.
"""

from __future__ import absolute_import, division, print_function
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out

from .base_ops import *
from .model_spec import ModelSpec


class Network(nn.Module):

    def __init__(self,
                 spec,
                 num_labels=100,
                 in_channels=3,
                 stem_out_channels=128,
                 num_stacks=3,
                 num_modules_per_stack=3,
                 momentum=0.997,
                 eps=1e-5,
                 tf_like=False):
        """
        Args:
            spec: ModelSpec from nasbench, or a tuple (adjacency matrix, ops)
            num_labels: Number of output labels.
            in_channels: Number of input image channels.
            stem_out_channels: Number of output stem channels. Other hidden channels are computed and depend on this
                number.
            num_stacks: Number of stacks, in every stacks the cells have the same number of channels.
            num_modules_per_stack: Number of cells per stack.
        """
        super(Network, self).__init__()

        if isinstance(spec, tuple):
            spec = ModelSpec(spec[0], spec[1])

        self.cell_indices = set()

        self.tf_like = tf_like
        self.layers = nn.ModuleList([])

        # initial stem convolution
        out_channels = stem_out_channels
        stem_conv = ConvBnRelu(
            in_channels, out_channels, 3, 1, 1, momentum=momentum, eps=eps)
        self.layers.append(stem_conv)

        # stacked cells
        in_channels = out_channels
        for stack_num in range(num_stacks):
            # downsample after every but the last cell
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)

                out_channels *= 2

            for module_num in range(num_modules_per_stack):
                cell = Cell(
                    spec,
                    in_channels,
                    out_channels,
                    momentum=momentum,
                    eps=eps)
                self.layers.append(cell)
                in_channels = out_channels

                self.cell_indices.add(len(self.layers) - 1)

        self.classifier = nn.Linear(out_channels, num_labels)

        self._initialize_weights()

    def forward(self, x, is_feat=False, preact=False):
        feat_list = []
        for _, layer in enumerate(self.layers):
            x = layer(x)
            feat_list.append(x)
        out = torch.mean(x, (2, 3))
        feat_list.append(out)
        out = self.classifier(out)

        if is_feat:
            return feat_list, out
        else:
            return out

    def forward_with_features(self, x):
        features = []
        for _, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, Cell):
                features.append(x)
        out = torch.mean(x, (2, 3))
        features.append(out)
        logits = self.classifier(out)
        return features, logits

    def forward_before_global_avg_pool(self, x, is_feat=False, preact=False):
        # feat_list = []
        for _, layer in enumerate(self.layers):
            x = layer(x)
            # feat_list.append(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.tf_like:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    torch.nn.init.normal_(
                        m.weight,
                        mean=0,
                        std=1.0 / torch.sqrt(torch.tensor(fan_in)))
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))

                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if self.tf_like:
                    torch.nn.init.xavier_uniform_(m.weight)
                else:
                    m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Cell(nn.Module):
    """
    Builds the model using the adjacency matrix and op labels specified. Channels
    control the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.
    """

    def __init__(self,
                 spec,
                 in_channels,
                 out_channels,
                 momentum=0.1,
                 eps=1e-5):
        super(Cell, self).__init__()

        self.dev_param = nn.Parameter(torch.empty(0))

        self.matrix = spec.matrix
        self.num_vertices = np.shape(self.matrix)[0]

        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = compute_vertex_channels(in_channels,
                                                       out_channels,
                                                       self.matrix)
        #self.vertex_channels = [in_channels] + [out_channels] * (self.num_vertices - 1)

        # operation for each node
        self.vertex_op = nn.ModuleList([Placeholder()])
        for t in range(1, self.num_vertices - 1):
            op = OP_MAP[spec.ops[t]](self.vertex_channels[t],
                                     self.vertex_channels[t])
            self.vertex_op.append(op)

        # operation for input on each vertex
        self.input_op = nn.ModuleList([Placeholder()])
        for t in range(1, self.num_vertices):
            if self.matrix[0, t]:
                self.input_op.append(
                    projection(
                        in_channels,
                        self.vertex_channels[t],
                        momentum=momentum,
                        eps=eps))
            else:
                self.input_op.append(Placeholder())

        self.last_inop: projection = self.input_op[self.num_vertices - 1]

    def forward(self, x):
        tensors = [x]

        out_concat = []
        # range(1, self.num_vertices - 1),
        for t, (inmod,
                outmod) in enumerate(zip(self.input_op, self.vertex_op)):
            if 0 < t < (self.num_vertices - 1):

                fan_in = []
                for src in range(1, t):
                    if self.matrix[src, t]:
                        fan_in.append(
                            truncate(tensors[src],
                                     torch.tensor(self.vertex_channels[t])))

                if self.matrix[0, t]:
                    l = inmod(x)
                    fan_in.append(l)

                # perform operation on node
                vertex_input = torch.zeros_like(fan_in[0]).to(
                    self.dev_param.device)
                for val in fan_in:
                    vertex_input += val

                vertex_output = outmod(vertex_input)

                tensors.append(vertex_output)
                if self.matrix[t, self.num_vertices - 1]:
                    out_concat.append(tensors[t])

        if not out_concat:
            assert self.matrix[0, self.num_vertices - 1]
            outputs = self.last_inop(tensors[0])
        else:
            if len(out_concat) == 1:
                outputs = out_concat[0]
            else:
                outputs = torch.cat(out_concat, 1)

            if self.matrix[0, self.num_vertices - 1]:
                outputs = outputs + self.last_inop(tensors[0])

        return outputs


def projection(in_channels, out_channels, momentum=0.1, eps=1e-5):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    return ConvBnRelu(in_channels, out_channels, 1, momentum=momentum, eps=eps)


def truncate(inputs, channels):
    """Slice the inputs to channels if necessary."""
    input_channels = inputs.size()[1]
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs  # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]


def compute_vertex_channels(in_channels, out_channels, matrix):
    """Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Code from https://github.com/google-research/nasbench/
    Returns:
        list of channel counts, in order of the vertices.
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()

    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v],
                                             vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return [int(v) for v in vertex_channels]


class Placeholder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return x
