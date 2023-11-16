from timm.data.transforms import (
    RandomResizedCropAndInterpolation as _RandomResizedCropAndInterpolation,
)
from timm.data import create_transform
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import json
import os
import random
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from iopath.common.file_io import PathManagerFactory
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

pathmgr = PathManagerFactory.get()

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RandomResizedCropAndInterpolation(_RandomResizedCropAndInterpolation):
    def __call__(self, img, features):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation

        out_img = F.resized_crop(img, i, j, h, w, self.size, interpolation)

        i, j, h, w = i / img.size[1], j / \
            img.size[0], h / img.size[1], w / img.size[0]
        out_feats = []
        for feat in features:
            feat_h, feat_w = feat.shape[-2:]
            feat = F.resized_crop(
                feat,
                int(i * feat_h),
                int(j * feat_w),
                int(h * feat_h),
                int(w * feat_w),
                size=(feat_h, feat_w),
            )
            out_feats.append(feat)

        return out_img, out_feats


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img, features):
        if torch.rand(1) < self.p:
            out_img = F.hflip(img)
            out_feats = []
            for feat in features:
                out_feats.append(F.hflip(feat))
            return out_img, out_feats
        return img, features


def create_train_transform(mean=None, std=None):
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std

    size = (224, 224)
    transform = create_transform(
        input_size=size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
        separate=True,
        mean=mean,
        std=std,
    )
    primary_tfl, secondary_tfl, final_tfl = transform
    return primary_tfl, secondary_tfl, final_tfl


def create_test_transform(mean=None, std=None):
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std

    primary_tfl = transforms.Resize((224, 224))
    secondary_tfl = transforms.Compose([])
    final_tfl = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD),
            ),
        ]
    )
    return primary_tfl, secondary_tfl, final_tfl


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for dataset with support for offline distillation.
    """

    def __init__(self, split):
        if split == 'train':
            transforms = create_train_transform()
        else:
            transforms = create_test_transform()
        self.primary_tfl, self.secondary_tfl, self.final_tfl = transforms

        self.features = None

    @abstractmethod
    def _get_data(self, index):
        """
        Returns the image and its label at index.
        """
        pass

    def __getitem__(self, index):
        img, label = self._get_data(index)
        if self.features:
            features = [torch.from_numpy(f[index].copy())
                        for f in self.features]
            for t in self.primary_tfl:
                img, features = t(img, features)
        else:
            img = self.primary_tfl(img)
            features = []

        img = self.secondary_tfl(img)
        img = self.final_tfl(img)

        return img, label, features


class Cifar100(BaseDataset):
    def __init__(self, data_path, split):
        super(Cifar100, self).__init__(split)
        assert pathmgr.exists(
            data_path), "Data path '{}' not found".format(data_path)
        splits = ['train', 'test']
        assert split in splits, "Split '{}' not supported for cifar".format(
            split)
        self.database = CIFAR100(
            root=data_path, train=split == 'train', download=True)

    def __len__(self):
        return len(self.database)

    def _get_data(self, index):
        return self.database[index]


class Chaoyang(BaseDataset):
    def __init__(self, data_path, split):
        super(Chaoyang, self).__init__(split)
        assert pathmgr.exists(
            data_path), "Data path '{}' not found".format(data_path)
        splits = ['train', 'test']
        assert split in splits, "Split '{}' not supported for Chaoyang".format(
            split)
        self.data_path = data_path
        with open(os.path.join(data_path, f'{split}.json'), 'r') as f:
            anns = json.load(f)
        self.data = anns

    def __len__(self):
        return len(self.data)

    def _get_data(self, index):
        ann = self.data[index]
        img = Image.open(os.path.join(self.data_path, ann['name']))
        return img, ann['label']
