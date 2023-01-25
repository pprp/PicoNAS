import numpy as np
import torch
from torchvision.datasets.imagenet import ImageFolder

from nanonas.datasets.build import build_dataloader


def get_random_data(name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if name == 'imagenet':
        # TODO
        train_loader = ImageFolder(
            root='~/data/imagenet',
            transform=None,
        )
    else:
        train_loader = build_dataloader(name, type='train')

    random_idxs = np.random.randint(
        0, len(train_loader.dataset), size=train_loader.batch_size)
    (more_data_X,
     more_data_y) = zip(*[train_loader.dataset[idx] for idx in random_idxs])
    more_data_X = torch.stack(more_data_X, dim=0).to(device)
    more_data_y = torch.Tensor(more_data_y).long().to(device)
    return more_data_X,
