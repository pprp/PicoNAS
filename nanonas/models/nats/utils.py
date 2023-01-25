import torch.nn as nn


def reset(m):
    # reset conv2d/linear/BN in Block
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        m.reset_parameters()
