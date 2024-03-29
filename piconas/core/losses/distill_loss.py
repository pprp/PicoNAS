import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CC(nn.Module):
    """
    Correlation Congruence for Knowledge Distillation
    http://openaccess.thecvf.com/content_ICCV_2019/papers/
    Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
    """

    def __init__(self, gamma=0.4, P_order=2):
        super(CC, self).__init__()
        self.gamma = gamma
        self.P_order = P_order

    def forward(self, feat_s, feat_t):
        corr_mat_s = self.get_correlation_matrix(feat_s)
        corr_mat_t = self.get_correlation_matrix(feat_t)

        return F.mse_loss(corr_mat_s, corr_mat_t)

    def get_correlation_matrix(self, feat):
        feat = F.normalize(feat, p=2, dim=-1)
        sim_mat = torch.matmul(feat, feat.t())
        corr_mat = torch.zeros_like(sim_mat)

        for p in range(self.P_order + 1):
            corr_mat += (
                math.exp(-2 * self.gamma)
                * (2 * self.gamma) ** p
                / math.factorial(p)
                * torch.pow(sim_mat, p)
            )

        return corr_mat


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, (
            f'KLDivergence supports reduction {accept_reduction}, '
            f'but gets {reduction}.'
        )
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if self.teacher_detach:
            preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction
        )
        return self.loss_weight * loss


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(
        a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps
    )


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    """https://github.com/hunto/image_classification_sota
    /blob/d4f15a0494/lib/models/losses/dist_kd.py
    """

    def __init__(self):
        super(DIST, self).__init__()

    def forward(
        self,
        z_s,
        z_t,
        beta: Union[float, Tensor] = 1.0,
        gamma: Union[float, Tensor] = 1.0,
    ):
        y_s = z_s.softmax(dim=1)
        y_t = z_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        if isinstance(beta, Tensor) and isinstance(gamma, Tensor):
            kd_loss = beta * inter_loss + gamma * intra_loss
        else:
            kd_loss = beta * inter_loss + gamma * intra_loss
        return kd_loss
