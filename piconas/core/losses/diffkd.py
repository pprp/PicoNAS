import torch


def diffkendall_2d(x, y, alpha=0.5):
    """
    Differentiable approximation of Kendall's rank correlation.

    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (N, D)
        alpha: Hyperparameter for sigmoid approximation

    Returns:
        diffkendall: Tensor of shape (N,)
    """

    N, D = x.shape

    # Pairwise difference
    x_diff = x[:, None, :] - x[:, :, None]  # (N, D, D)
    y_diff = y[:, None, :] - y[:, :, None]

    # Numerator
    num = (torch.sigmoid(alpha*x_diff) - torch.sigmoid(-alpha*x_diff)) * \
          (torch.sigmoid(alpha*y_diff) - torch.sigmoid(-alpha*y_diff))

    # Denominator
    den = (torch.sigmoid(alpha*x_diff) + torch.sigmoid(-alpha*x_diff)) * \
          (torch.sigmoid(alpha*y_diff) + torch.sigmoid(-alpha*y_diff))

    # DiffKendall
    diffkendall = (num.sum(-1).sum(-1) - num.diagonal(dim1=-2, dim2=-1).sum(-1)) / \
                  ((D-1) * D / 2)

    return diffkendall


def diffkendall(x, y, alpha=0.5, beta=1.0):
    """
    Differentiable approximation of Kendall's rank correlation.

    Args:
        x: Tensor of shape [N]
        y: Tensor of shape [N]
        alpha: Hyperparameter for sigmoid approximation
        beta: Scaling factor for the sigmoid function

    Returns:
        diffkendall: Scalar value representing the correlation
    """

    # Ensure x and y are 1D tensors
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise ValueError('x and y should be 1-dimensional tensors')

    N = x.shape[0]

    # Pairwise difference
    x_diff = x[:, None] - x[None, :]  # Shape: (N, N)
    y_diff = y[:, None] - y[None, :]  # Shape: (N, N)

    # Numerator
    num = (torch.sigmoid(beta * alpha * x_diff) - torch.sigmoid(-beta * alpha * x_diff)) * \
          (torch.sigmoid(beta * alpha * y_diff) - torch.sigmoid(-beta * alpha * y_diff))

    # Exclude diagonal elements from the numerator
    num = num - torch.diag(torch.diag(num))

    # Denominator
    denom = N * (N - 1) / 2

    # DiffKendall
    diffkendall = torch.tensor(-1) * num.sum() / denom

    return diffkendall
