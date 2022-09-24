#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""NAS network (adopted from DARTS)."""

import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

from .genotypes import GENOTYPES
from .ops import OPS, FactorizedReduce, Identity, ReLUConvBN

logger = logging.get_logger(__name__)


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, rates):
        super(ASPP, self).__init__()
        assert len(rates) in [1, 3]
        self.rates = rates
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                dilation=rates[0],
                padding=rates[0],
                bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        if len(self.rates) == 3:
            self.aspp3 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    dilation=rates[1],
                    padding=rates[1],
                    bias=False), nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
            self.aspp4 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    dilation=rates[2],
                    padding=rates[2],
                    bias=False), nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        self.aspp5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Conv2d(
                out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1))

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]),
                         mode='bilinear',
                         align_corners=True)(
                             x5)
        if len(self.rates) == 3:
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x = torch.cat((x1, x2, x3, x4, x5), 1)
        else:
            x = torch.cat((x1, x2, x5), 1)
        x = self.classifier(x)
        return x


class Classifier(nn.Module):

    def __init__(self, channels, num_classes):
        super(Classifier, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x, shape):
        x = self.pooling(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


def drop_path(x, drop_prob):
    """Drop path (ported from DARTS)."""
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(
            torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):
    """NAS cell (ported from DARTS)."""

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,
                 reduction_prev):
        super(Cell, self).__init__()
        logger.info('{}, {}, {}'.format(C_prev_prev, C_prev, C))

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):
    """CIFAR network (ported from DARTS)."""

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary,
                                                     num_classes)
        self.classifier = Classifier(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        logits = self.classifier(s1, input.shape[2:])
        if self._auxiliary and self.training:
            return logits, logits_aux
        return logits


class NetworkImageNet(nn.Module):
    """ImageNet network (ported from DARTS)."""

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(
                3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        reduction_layers = [layers // 3, 2 * layers // 3]
        for i in range(layers):
            if i in reduction_layers:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(
                C_to_auxiliary, num_classes)
        self.classifier = Classifier(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        logits = self.classifier(s1, input.shape[2:])
        if self._auxiliary and self.training:
            return logits, logits_aux
        return logits


class NAS(nn.Module):
    """NAS net wrapper (delegates to nets from DARTS)."""

    def __init__(self, dataset='cifar10', genotype='nas'):
        assert dataset in ['cifar10', 'imagenet', 'cityscapes'], \
            'Training on {} is not supported'.format(dataset)
        assert dataset in ['cifar10', 'imagenet', 'cityscapes'], \
            'Testing on {} is not supported'.format(dataset)

        assert genotype in GENOTYPES, \
            'Genotype {} not supported'.format(genotype)
        super(NAS, self).__init__()

        # Use a custom or predefined genotype
        genotype = GENOTYPES[genotype]

        # Determine the network constructor for dataset
        if 'cifar' in dataset:
            net_ctor = NetworkCIFAR
        else:
            net_ctor = NetworkImageNet
        # Construct the network
        self.net_ = net_ctor(
            C=16,
            num_classes=10,
            layers=20,
            auxiliary=False,
            genotype=genotype)
        # Drop path probability (set / annealed based on epoch)
        self.net_.drop_path_prob = 0.0

    def set_drop_path_prob(self, drop_path_prob):
        self.net_.drop_path_prob = drop_path_prob

    def forward(self, x):
        return self.net_.forward(x)
