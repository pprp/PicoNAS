from __future__ import print_function
import os

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from .samplers import get_proxy_data_log_entropy_histogram, read_entropy_file


def get_cifar100_dataloaders_entropy(batch_size=128,
                                     train_portion=0.5,
                                     sampling_portion=0.2,
                                     num_workers=8,
                                     k=4096,
                                     mode='exact',
                                     is_sample=True,
                                     percent=1.0,
                                     save='./tmp',
                                     num_class=100):
    """
    cifar 100
    """

    entropy_file = './dataset/entropy_list/cifar100_resnet56_index_entropy_class.txt'
    index, entropy, label = read_entropy_file(entropy_file)

    indices = get_proxy_data_log_entropy_histogram(
        entropy,
        sampling_portion=sampling_portion,
        sampling_type=1,
        dataset='cifar100')

    num_train = num_proxy_data = len(indices)
    split = int(np.floor(train_portion * num_proxy_data))

    num_classes = [0] * num_class

    if not os.path.exists(save):
        # make a template dir to save the entropy file.
        os.makedirs(save)

    with open(os.path.join(save, 'proxy_train_entropy_file.txt'), 'w') as f:
        for idx in indices[:split]:
            f.write('%d %f %d\n' % (idx, entropy[idx], label[idx]))
            num_classes[label[idx]] += 1
    with open(os.path.join(save, 'proxy_val_entropy_file.txt'), 'w') as f:
        for idx in indices[split:num_train]:
            f.write('%d %f %d\n' % (idx, entropy[idx], label[idx]))
            num_classes[label[idx]] += 1

    data_folder = './data/cifar'

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(
        root=data_folder, download=True, train=True, transform=train_transform)

    n_data = len(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,  # need to be False when use sampler
        num_workers=num_workers,
        pin_memory=True,
        sampler=SubsetRandomSampler(indices[:split]))

    valid_loader = DataLoader(
        train_set,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(num_workers / 2),
        pin_memory=True,
        sampler=SubsetRandomSampler(indices[split:num_train]))

    return train_loader, valid_loader, n_data
