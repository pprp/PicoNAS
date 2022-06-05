import unittest
from unittest import TestCase

from pplib.datasets import build_dataloader, build_dataset


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattribute__ = dict.__getitem__


class TestDataset(TestCase):

    def test_dataset(self):
        args = dict(
            bs=64,
            data_dir='./data/cifar',
            fast=False,
            nw=2,
            random_erase=None,
            autoaugmentation=None,
            cutout=None,
        )
        dataset = build_dataset(
            type='train', name='cifar10', config=Dict(args))
        assert dataset is not None
        for i, (img, label) in enumerate(dataset):
            if i > 10:
                break
            print(img.shape, label)

    def test_dataloader(self):
        args = dict(
            bs=64,
            fast=False,
            nw=2,
            random_erase=None,
            autoaugmentation=None,
            cutout=None,
            data_dir='./data/cifar')
        dataloader = build_dataloader(config=Dict(args))
        assert dataloader is not None

        for i, (img, label) in enumerate(dataloader):
            if i > 10:
                break
            print(img.shape, label.shape)


if __name__ == '__main__':
    unittest.main()
