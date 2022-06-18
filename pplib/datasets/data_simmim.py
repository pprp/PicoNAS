# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import numpy as np
import torch  # noqa: F401
import torch.distributed as dist  # noqa: F401
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # noqa: F401
from torch.utils.data import DataLoader, DistributedSampler  # noqa: F401
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder  # noqa: F401


class MaskGenerator:
    """Generate mask with ratio"""

    def __init__(self,
                 input_size=32,
                 mask_patch_size=2,
                 model_patch_size=2,
                 mask_ratio=0.6):

        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SimMIMTransform:

    def __init__(self):
        self.transform_img = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.mask_generator = MaskGenerator(
            input_size=32,
            mask_patch_size=2,
            model_patch_size=2,  # for vit
            mask_ratio=0.6,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(
                    default_collate(
                        [batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(logger):
    transform = SimMIMTransform()
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = datasets.CIFAR10(
        root='./data/cifar',
        train=True,
        download=True,
        transform=transform,
    )
    logger.info(f'Build dataset: train images = {len(dataset)}')

    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=dist.get_rank(),
    #     shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        # sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    return dataloader
