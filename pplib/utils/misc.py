from typing import Dict

import numpy as np
import torch
from torch import Tensor


def convert_arch2dict(arch: str) -> Dict:
    """Convert the arch encoding to subnet config.

    Args:
        arch (str): arch config with 14 chars.

    Returns:
        Dict: subnet config.
    """
    assert len(arch) == 14

    specific_subnet = {}
    for c, id in zip(arch, list(range(14))):
        specific_subnet[id] = c
    return specific_subnet


def convertTensor2Pltimg(img: Tensor) -> np.array:
    try:
        img = img.numpy()
    except TypeError:
        img = img.cpu()
        img = img.detach().numpy()

    img = np.transpose(img, (1, 2, 0))

    return img


def convertTensor2BoardImage(img):
    img = img.cpu().detach()
    # convert to 0-1
    _, H, W = img.shape
    mean = Tensor(np.array([0.485, 0.456, 0.406
                            ])).unsqueeze(-1).unsqueeze(-1).repeat(1, H, W)
    std = Tensor(np.array([0.229, 0.224, 0.225
                           ])).unsqueeze(-1).unsqueeze(-1).repeat(1, H, W)
    img = std * img + mean

    img = torch.clip(img, 0, 1)
    return img
