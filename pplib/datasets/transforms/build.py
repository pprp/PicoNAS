import torchvision.transforms as transforms

from .autoaugment import CIFAR10Policy
from .cutout import Cutout
from .randomerase import RandomErase
from .simmim_transform import SimMIMTransform


def build_transforms(dataset='cifar10', type='train', config=None):
    assert type in ['train', 'val']
    assert dataset in ['cifar10', 'cifar100', 'simmim']
    transform_type = None

    if type == 'train':
        base_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if getattr(config, 'random_erase', False):
            mid_transform = [
                RandomErase(
                    config.random_erase_prob,
                    config.random_erase_sl,
                    config.random_erase_sh,
                    config.random_erase_r,
                ),
            ]
        elif getattr(config, 'autoaugmentation', False):
            mid_transform = [
                CIFAR10Policy(),
            ]
        else:
            mid_transform = []

        if dataset == 'cifar10':
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]
        elif dataset == 'cifar100':
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4865, 0.4409],
                                     [0.1942, 0.1918, 0.1958]),
            ]
        elif dataset == 'simmim':
            post_transform = [SimMIMTransform()]

        if getattr(config, 'cutout', False):
            post_transform.append(Cutout(1, 8))

        transform_type = transforms.Compose(
            [*base_transform, *mid_transform, *post_transform])

    elif type == 'val':
        if dataset == 'cifar10':
            transform_type = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        elif dataset == 'cifar100':
            transform_type = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4865, 0.4409],
                                     [0.1942, 0.1918, 0.1958]),
            ])
        elif dataset == 'simmim':
            transform_type = SimMIMTransform()

    else:
        raise 'Type Error in transforms'

    return transform_type
