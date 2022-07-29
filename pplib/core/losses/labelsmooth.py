# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation adapted from AlphaNet - https://github.com/facebookresearch/AlphaNet/blob/main/loss_ops.py  # noqa: E501

import torch


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):

    def __init__(self, label_smoothing=0.1):
        super(CrossEntropyLossSmooth, self).__init__()
        self.eps = label_smoothing

    """ label smooth """

    def forward(self, output, target):
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - self.eps) + self.eps / n_class
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        loss = -torch.bmm(target, output_log_prob)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
