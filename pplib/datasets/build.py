import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from .transforms import build_transforms


def build_dataset(type='train', name='cifar10', config=None, fast=False):
    assert name in ['cifar10', 'cifar100']
    assert type in ['train', 'val']

    dataset_type = None

    if name == 'cifar10':
        if type == 'train':
            dataset_type = datasets.CIFAR10(
                root=config.data_dir,
                train=True,
                download=True,
                transform=build_transforms('cifar10', 'train', config=config),
            )
        elif type == 'val':
            dataset_type = datasets.CIFAR10(
                root=config.data_dir,
                train=False,
                download=True,
                transform=build_transforms('cifar10', 'val', config=config),
            )

    elif name == 'cifar100':
        if type == 'train':
            dataset_type = datasets.CIFAR100(
                root=config.data_dir,
                train=True,
                download=True,
                transform=build_transforms('cifar10', 'train', config=config),
            )
        elif type == 'val':
            dataset_type = datasets.CIFAR100(
                root=config.data_dir,
                train=False,
                download=True,
                transform=build_transforms('cifar10', 'val', config=config),
            )
    else:
        raise 'Type Error: {} Not Supported'.format(name)

    if fast:
        # fast train using ratio% images
        ratio = 0.3
        total_num = len(dataset_type.targets)
        choice_num = int(total_num * ratio)
        print(f'FAST MODE: Choice num/Total num: {choice_num}/{total_num}')

        dataset_type.data = dataset_type.data[:choice_num]
        dataset_type.targets = dataset_type.targets[:choice_num]

    print('DATASET:', len(dataset_type))

    return dataset_type


def build_dataloader(name='cifar10', type='train', config=None):
    assert type in ['train', 'val']
    assert name in ['cifar10', 'cifar100']
    if name == 'cifar10':
        if type == 'train':
            dataloader_type = DataLoader(
                build_dataset(
                    'train', 'cifar10', config=config, fast=config.fast),
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.nw,
                pin_memory=True,
            )
        elif type == 'val':
            dataloader_type = DataLoader(
                build_dataset(
                    'val', 'cifar10', config=config, fast=config.fast),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.nw,
                pin_memory=True,
            )
    elif name == 'cifar100':
        if type == 'train':
            dataloader_type = DataLoader(
                build_dataset(
                    'train', 'cifar100', config=config, fast=config.fast),
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.nw,
                pin_memory=True,
            )
        elif type == 'val':
            dataloader_type = DataLoader(
                build_dataset(
                    'val', 'cifar100', config=config, fast=config.fast),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.nw,
                pin_memory=True,
            )
    else:
        raise 'Type Error: {} Not Supported'.format(name)

    return dataloader_type
