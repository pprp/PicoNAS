import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .imagenet16 import ImageNet16
from .transforms import build_transforms


def collate_fn(batch):
    """collate function for simmim."""
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
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


def build_dataset(type='train', dataset='cifar10', config=None, fast=False):
    assert dataset in ['cifar10', 'cifar100', 'simmim', 'ImageNet16-120']
    assert type in ['train', 'val']

    dataset_type = None

    if config:
        data_dir = getattr(config, 'data_dir', './data/cifar')
    else:
        data_dir = './data/cifar'

    if dataset == 'cifar10':
        if type == 'train':
            dataset_type = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=build_transforms('cifar10', 'train', config=config),
            )
        elif type == 'val':
            dataset_type = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=build_transforms('cifar10', 'val', config=config),
            )

    elif dataset == 'cifar100':
        if type == 'train':
            dataset_type = datasets.CIFAR100(
                root=data_dir,
                train=True,
                download=True,
                transform=build_transforms('cifar10', 'train', config=config),
            )
        elif type == 'val':
            dataset_type = datasets.CIFAR100(
                root=data_dir,
                train=False,
                download=True,
                transform=build_transforms('cifar10', 'val', config=config),
            )
    elif dataset == 'simmim':
        if type == 'train':
            dataset_type = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=build_transforms('simmim', 'train', config=config),
            )
        elif type == 'val':
            dataset_type = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=build_transforms('simmim', 'val', config=config),
            )
    elif dataset == 'ImageNet16-120':
        if type == 'train':
            dataset_type = ImageNet16(
                root=data_dir,
                train=True,
                transform=build_transforms(
                    'ImageNet16-120', 'train', config=config),
                use_num_of_class_only=120)
        elif type == 'val':
            dataset_type = ImageNet16(
                root=data_dir,
                train=False,
                transform=build_transforms(
                    'ImageNet16-120', 'val', config=config),
                use_num_of_class_only=120)
    else:
        raise f'Type Error: {dataset} Not Supported'

    if fast:
        process_fast(dataset_type)
    print('DATASET:', len(dataset_type))

    return dataset_type


# TODO Rename this here and in `build_dataset`
def process_fast(dataset_type):
    # fast train using ratio% images
    ratio = 0.3
    total_num = len(dataset_type.targets)
    choice_num = int(total_num * ratio)
    print(f'FAST MODE: Choice num/Total num: {choice_num}/{total_num}')

    dataset_type.data = dataset_type.data[:choice_num]
    dataset_type.targets = dataset_type.targets[:choice_num]


def build_dataloader(dataset='cifar10', type='train', config=None):
    assert type in ['train', 'val']
    assert dataset in ['cifar10', 'cifar100', 'simmim', 'ImageNet16-120']

    if config:
        batch_size = getattr(config, 'batch_size', 64)
        nw = getattr(config, 'nw', 4)
        fast = getattr(config, 'fast', False)
    else:
        batch_size = 1
        nw = 0
        fast = False

    if dataset == 'cifar10':
        if type == 'train':
            dataloader_type = DataLoader(
                build_dataset('train', 'cifar10', config=config, fast=fast),
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                pin_memory=True,
            )
        elif type == 'val':
            dataloader_type = DataLoader(
                build_dataset('val', 'cifar10', config=config, fast=fast),
                batch_size=batch_size,
                shuffle=False,
                num_workers=nw,
                pin_memory=True,
            )
    elif dataset == 'cifar100':
        if type == 'train':
            dataloader_type = DataLoader(
                build_dataset('train', 'cifar100', config=config, fast=fast),
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                pin_memory=True,
            )
        elif type == 'val':
            dataloader_type = DataLoader(
                build_dataset('val', 'cifar100', config=config, fast=fast),
                batch_size=batch_size,
                shuffle=False,
                num_workers=nw,
                pin_memory=True,
            )
    elif dataset == 'simmim':
        if type == 'train':
            dataloader_type = DataLoader(
                build_dataset('train', 'simmim', config=config, fast=fast),
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        elif type == 'val':
            dataloader_type = DataLoader(
                build_dataset('val', 'simmim', config=config, fast=fast),
                batch_size=batch_size,
                shuffle=False,
                num_workers=nw,
                pin_memory=True,
                collate_fn=collate_fn,
            )
    elif dataset == 'ImageNet16-120':
        if type == 'train':
            dataloader_type = DataLoader(
                build_dataset(
                    'train', 'ImageNet16-120', config=config, fast=fast),
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                pin_memory=True,
            )
        elif type == 'val':
            dataloader_type = DataLoader(
                build_dataset(
                    'val', 'ImageNet16-120', config=config, fast=fast),
                batch_size=batch_size,
                shuffle=False,
                num_workers=nw,
                pin_memory=True,
            )
    else:
        raise f'Type Error: {dataset} Not Supported'

    return dataloader_type
