import unittest
from unittest import TestCase

from nb201_datasets import get_datasets, get_nas_search_loaders


class TestDataloader(TestCase):
    def test_cifar10(self):
        """
        name: cifar10, cifar100, ImageNet16-120
        """
        train_data, valid_data, xshape, class_num = get_datasets(
            name='cifar10', root='../../data/cifar', cutout=-1
        )
        search_loader, _, valid_loader = get_nas_search_loaders(
            train_data,
            valid_data,
            dataset='cifar10',
            config_root='./config/',
            batch_size=(512, 512),
            workers=2,
        )
        for timg, tlabel, vimg, vlabel in search_loader:
            print(timg.shape, tlabel.shape, vimg.shape, vlabel.shape)
            break

    def test_cifar100(self):
        train_data, valid_data, xshape, class_num = get_datasets(
            name='cifar100', root='../../data/cifar', cutout=-1
        )
        search_loader, _, valid_loader = get_nas_search_loaders(
            train_data,
            valid_data,
            dataset='cifar100',
            config_root='./config/',
            batch_size=(512, 512),
            workers=2,
        )
        for timg, tlabel, vimg, vlabel in search_loader:
            print(timg.shape, tlabel.shape, vimg.shape, vlabel.shape)
            break

    def test_imagenet16(self):
        train_data, valid_data, xshape, class_num = get_datasets(
            name='ImageNet16-120', root='../../data/ImageNet16', cutout=-1
        )
        search_loader, _, valid_loader = get_nas_search_loaders(
            train_data,
            valid_data,
            dataset='ImageNet16-120',
            config_root='./config/',
            batch_size=(512, 512),
            workers=2,
        )
        for timg, tlabel, vimg, vlabel in search_loader:
            print(timg.shape, tlabel.shape, vimg.shape, vlabel.shape)
            break


if __name__ == '__main__':
    unittest.main()
