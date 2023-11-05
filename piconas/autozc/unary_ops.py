import random
from typing import TypeVar, Union

import torch
import torch.nn.functional as F

Scalar = TypeVar('Scalar')
Vector = TypeVar('Vector')
Matrix = TypeVar('Matrix')

ALLTYPE = Union[Union[Scalar, Vector], Matrix]

UNARY_KEYS = [
    'element_wise_log',
    'element_wise_abslog',
    'element_wise_abs',
    'element_wise_pow',
    'element_wise_exp',
    'normalize',
    'element_wise_relu',
    'element_wise_sign',
    'element_wise_invert',
    'frobenius_norm',
    'element_wise_normalized_sum',
    'l1_norm',
    'softmax',
    'sigmoid',
    'slogdet',
    'logsoftmax',
    'to_mean_scalar',
    'to_sum_scalar',
    'gram_matrix',
    'no_op',
]

# unary operation


def no_op(A: ALLTYPE) -> ALLTYPE:
    return A


def element_wise_log(A: ALLTYPE) -> ALLTYPE:
    A[A <= 0] == 1
    return torch.log(A)


def element_wise_revert(A: ALLTYPE) -> ALLTYPE:
    return A * -1


def element_wise_abslog(A: ALLTYPE) -> ALLTYPE:
    A[A == 0] = 1
    A = torch.abs(A)
    return torch.log(A)


def element_wise_abs(A: ALLTYPE) -> ALLTYPE:
    return torch.abs(A)


def element_wise_pow(A: ALLTYPE) -> ALLTYPE:
    return torch.pow(A, 2)


def element_wise_exp(A: ALLTYPE) -> ALLTYPE:
    return torch.exp(A)


def normalize(A: ALLTYPE) -> ALLTYPE:
    m = torch.mean(A)
    s = torch.std(A)
    C = (A - m) / s
    C[C != C] = 0
    return C


def element_wise_relu(A: ALLTYPE) -> ALLTYPE:
    return F.relu(A)


def element_wise_sign(A: ALLTYPE) -> ALLTYPE:
    return torch.sign(A)


def element_wise_invert(A: ALLTYPE) -> ALLTYPE:
    if isinstance(A, (int, float)) and A == 0:
        raise ZeroDivisionError
    return 1 / (A + 1e-9)


def frobenius_norm(A: ALLTYPE) -> Scalar:
    return torch.norm(A, p='fro')


def element_wise_normalized_sum(A: ALLTYPE) -> Scalar:
    return torch.sum(A) / A.numel()


def l1_norm(A: ALLTYPE) -> Scalar:
    return torch.sum(torch.abs(A)) / A.numel()


def p_dist(A: Matrix) -> Vector:
    return F.pdist(A)


def softmax(A: ALLTYPE) -> ALLTYPE:
    return F.softmax(A, dim=0)


def logsoftmax(A: ALLTYPE) -> ALLTYPE:
    return F.log_softmax(A)


def sigmoid(A: ALLTYPE) -> ALLTYPE:
    return torch.sigmoid(A)


def slogdet(A: Matrix) -> Scalar:
    sign, value = torch.linalg.slogdet(A)
    return value


def to_mean_scalar(A: ALLTYPE) -> Scalar:
    return torch.mean(A)


def to_sum_scalar(A: ALLTYPE) -> Scalar:
    return torch.sum(A)


def gram_matrix(A: Matrix) -> Matrix:
    """https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """
    a, b, c, d = A.size()
    feature = A.view(a * b, c * d)
    G = torch.mm(feature, feature.t())
    return G.div(a * b * c * d)


def unary_operation(A, idx=None):
    if idx is None:
        idx = random.choice(range(len(UNARY_KEYS)))

    assert idx < len(UNARY_KEYS)

    unaries = {
        'element_wise_log': element_wise_log,
        'element_wise_abslog': element_wise_abslog,
        'element_wise_abs': element_wise_abs,
        'element_wise_pow': element_wise_pow,
        'element_wise_exp': element_wise_exp,
        'normalize': normalize,
        'element_wise_relu': element_wise_relu,
        'element_wise_sign': element_wise_sign,
        'element_wise_invert': element_wise_invert,
        'frobenius_norm': frobenius_norm,
        'element_wise_normalized_sum': element_wise_normalized_sum,
        'l1_norm': l1_norm,
        'softmax': softmax,
        'sigmoid': sigmoid,
        'slogdet': slogdet,
        'p_dist': p_dist,
        'to_mean_scalar': to_mean_scalar,
        'to_sum_scalar': to_sum_scalar,
        'gram_matrix': gram_matrix,
        'logsoftmax': logsoftmax,
        'no_op': no_op,
    }
    return unaries[UNARY_KEYS[idx]](A)
