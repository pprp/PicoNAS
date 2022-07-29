# https://github.com/kcyu2014/nas-landmarkreg

import torch
import torch.nn.functional as F


def rank_cross_entropy_loss(l1, l2):
    logit = torch.sigmoid(l1 - l2)
    # logit = F.prelu(logit, torch.tensor(0.1))
    return -torch.log(1 - logit)


def rank_infinite_loss_v1(l1, l2, w1, w2):
    d = (l1 - l2) / (w1 - w2)
    # in this case: l1 < l2 and w1 < w2, or l1 > l2
    #     and w1 > w2, d > 0. but we should
    p = torch.sigmoid(-d)
    return F.relu(0.5 - p)


def rank_infinite_loss_v2(l1, l2, w1, w2):
    d = (l1 - l2) / (w1 - w2)
    p = torch.sigmoid(-d)
    return F.softplus(0.5 - p)


def rank_infinite_relu(l1, l2, w1, w2):
    d = (l1 - l2) * (w1 - w2)
    return F.relu(d)


def rank_infinite_softplus(l1, l2, w1, w2):
    d = (l1 - l2) * (w1 - w2)
    return F.softplus(d, beta=5)


def rank_hinge_sign_infinite(l1, l2, w1, w2):
    return F.relu(1 - (l1 - l2) * torch.sign(w2 - w1))


def rank_cross_entropy_focal_loss(l1, l2, gamma=5):
    logit = torch.sigmoid(l1 - l2)
    # logit = F.prelu(logit, torch.tensor(0.1))
    return -(logit).pow(gamma) * torch.log(1 - logit)


def rank_mixed_cross_entropy_loss(l1, l2):
    if l1 < l2:
        return rank_cross_entropy_focal_loss(l1, l2)
    else:
        return rank_cross_entropy_loss(l1, l2)


def tanh_sign_infinite(l1, l2, w1, w2):
    # given the fact that, l1 < l2 == w1 > w2.
    return F.relu(torch.tanh(l1 - l2) * torch.sign(w1 - w2))


def tanh_infinite(l1, l2, w1, w2):
    # given the fact that, l1 < l2 == w1 > w2.
    return F.relu(torch.tanh(l1 - l2) * torch.tanh(w1 - w2))


def tanh_infinite_norelu(l1, l2, w1, w2):
    # given the fact that, l1 < l2 == w1 > w2.
    return torch.tanh(l1 - l2) * torch.tanh(w1 - w2)


_loss_fn = {
    'mae_relu':
    lambda l1, l2: F.relu(l1 - l2),
    'mae_relu_inverse':
    lambda l1, l2: F.relu(l2 - l1),
    'mae_sign_relu':
    lambda l1, l2: F.relu(torch.sign(l1 - l2)),
    'mae_sign_tanh_relu':
    lambda l1, l2: F.relu(torch.sign(torch.tanh(l1 - l2))),
    'mae_tanh_relu':
    lambda l1, l2: F.relu(torch.tanh(l1 - l2)),
    'mae_softplus':
    lambda l1, l2: F.softplus(l1 - l2),
    'mae_softplus_beta3':
    lambda l1, l2: F.softplus(l1 - l2, beta=3),
    'mae_softplus_beta5':
    lambda l1, l2: F.softplus(l1 - l2, beta=5),
    'mae_softplus_beta7':
    lambda l1, l2: F.softplus(l1 - l2, beta=7),
    'focal_loss':
    rank_cross_entropy_focal_loss,
    'mae_relu_norm':
    lambda l1, l2: F.relu((l1 - l2) / (l1 - l2).abs() * (l1 + l2) / 2),
    'mae_tanh_infinite':
    tanh_infinite,
    'tanh_infinite':
    tanh_infinite_norelu,
    'mae_sign_tanh_infinite':
    tanh_sign_infinite,
    'mae_relu_sigmoid_infinite':
    rank_infinite_loss_v1,
    'mae_relu_infinite':
    rank_infinite_relu,
    'softplus_infinite':
    rank_infinite_softplus,
    'sigmoid_softplus_infinite':
    rank_infinite_loss_v2,
    'hinge_sign_infinite':
    rank_hinge_sign_infinite,
    'crossentropy':
    rank_cross_entropy_loss,
    'mixed_focal':
    rank_mixed_cross_entropy_loss,
}


def get_rank_loss_fn(name, weighted):
    """
    All of these loss will penalize l1 > l2,
        i.e. ground truth label is l1 < l2.
    :param name: args.landmark_loss_fn
    :param weighted: weighted to add a subscript.
    :return: loss fn.
    """

    if weighted == 'embed':
        return lambda l1, l2, w: w * _loss_fn[name](l1, l2)
    elif weighted == 'infinite':
        return _loss_fn[name + '_infinite']
    return _loss_fn[name]
