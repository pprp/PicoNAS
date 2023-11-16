import torch

from .build import build_criterion  # noqa: F401
from .diffkd import diffkendall  # noqa: F401,F403
from .distill_loss import KLDivergence  # noqa: F401
from .distill_loss import CC, DIST  # noqa: F401
from .landmark_loss import *  # noqa: F401,F403
from .rmi_loss import RMI_loss  # noqa: F401,F403


def pair_loss(outputs, labels):
    if len(labels.shape) == 0:  # Check if labels is a scalar
        labels = labels.unsqueeze(0)

    output = outputs.unsqueeze(1)

    repeat_dims = [1] * len(labels.shape)
    repeat_dims[-1] = labels.shape[0]

    label1 = labels.repeat(*repeat_dims)

    tmp = (output - output.t()) * torch.sign(label1 - label1.t())
    tmp = torch.log(1 + torch.exp(-tmp))
    eye_tmp = tmp * torch.eye(len(tmp)).cuda()
    new_tmp = tmp - eye_tmp
    loss = torch.sum(new_tmp) / (outputs.shape[0] * (outputs.shape[0] - 1))
    return loss


# def pair_loss(outputs, labels):
#     output = outputs.unsqueeze(1)
#     output1 = output.repeat(1, outputs.shape[0])
#     label = labels.unsqueeze(1)
#     label1 = label.repeat(1, labels.shape[0])
#     tmp = (output1 - output1.t()) * torch.sign(label1 - label1.t())
#     tmp = torch.log(1 + torch.exp(-tmp))
#     eye_tmp = tmp * torch.eye(len(tmp)).cuda()
#     new_tmp = tmp - eye_tmp
#     loss = torch.sum(new_tmp) / (outputs.shape[0] * (outputs.shape[0] - 1))
#     return loss


def pair_loss_with_prior(outputs, labels, priors, alpha=0.5):
    """
    Args:
        outputs (torch.Tensor): Model predictions.
        labels (torch.Tensor): True labels.
        priors (torch.Tensor): Prior information about the predictions.
        alpha (float): Weighting factor to balance the contribution of priors.
                       Value between 0 and 1; 0 means no contribution of priors,
                       and 1 means only priors contribute.

    Returns:
        loss (torch.Tensor): Pairwise ranking loss with prior information.
    """
    output = outputs.unsqueeze(1)
    output1 = output.repeat(1, outputs.shape[0])

    prior = priors.unsqueeze(1)
    prior1 = prior.repeat(1, priors.shape[0])

    label = labels.unsqueeze(1)
    label1 = label.repeat(1, labels.shape[0])

    # Calculate pairwise differences incorporating the priors
    pairwise_diff = (1 - alpha) * (output1 - output1.t()) + alpha * (
        prior1 - prior1.t()
    )

    # Calculate pairwise ranking using the modified differences
    tmp = pairwise_diff * torch.sign(label1 - label1.t())
    tmp = torch.log(1 + torch.exp(-tmp))

    eye_tmp = tmp * torch.eye(len(tmp)).cuda()
    new_tmp = tmp - eye_tmp

    loss = torch.sum(new_tmp) / (outputs.shape[0] * (outputs.shape[0] - 1))
    return loss
