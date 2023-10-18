##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import hashlib
import json
import os
import os.path as osp
import random
import sys
from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def convert_param(original_lists):
    support_types = ('str', 'int', 'bool', 'float', 'none')

    assert isinstance(
        original_lists,
        list), 'The type is not right : {:}'.format(original_lists)
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, 'Ctype={:}, support={:}'.format(
        ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    outs = []
    for x in value:
        if ctype == 'int':
            x = int(x)
        elif ctype == 'str':
            x = str(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            assert x == 'None', 'for none type, the value must be None instead of {:}'.format(
                x)
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
    if not is_list:
        outs = outs[0]
    return outs


def load_config(path, extra, logger):
    path = str(path)
    if hasattr(logger, 'log'):
        logger.log(path)
    assert os.path.exists(path), 'Can not find {:}'.format(path)
    # Reading data back
    with open(path, 'r') as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}
    assert extra is None or isinstance(
        extra, dict), 'invalid type of extra : {:}'.format(extra)
    if isinstance(extra, dict):
        content = {**content, **extra}
    Arguments = namedtuple('Configure', ' '.join(content.keys()))
    content = Arguments(**content)
    if hasattr(logger, 'log'):
        logger.log('{:}'.format(content))
    return content


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    return True if md5 is None else check_md5(fpath, md5)


class ImageNet16(data.Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
        ['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
        ['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
        ['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
        ['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
        ['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
        ['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
        ['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
        ['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
        ['train_data_batch_10', '8f03f34ac4b42271a294f91bf480f29b'],
    ]
    valid_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]

    def __init__(self, root, train, transform, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            # print ('Load {:}/{:02d}-th : {:}'.format(i, len(downloaded_list), file_path))
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if use_num_of_class_only is not None:
            assert isinstance(
                use_num_of_class_only, int
            ) and use_num_of_class_only > 0 and use_num_of_class_only < 1000, 'invalid use_num_of_class_only : {:}'.format(
                use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index] - 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.valid_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


class SearchDataset(data.Dataset):
    """_summary_

    Args:
        name (_type_): _description_
        data (_type_): _description_
        train_split (_type_): _description_
        valid_split (_type_): _description_
        check (bool, optional): _description_. Defaults to True.
    """

    def __init__(self, name, data, train_split, valid_split, check=True):

        self.datasetname = name
        if isinstance(data, (list, tuple)):  # new type of SearchDataset
            assert len(data) == 2, 'invalid length: {:}'.format(len(data))
            self.train_data = data[0]
            self.valid_data = data[1]
            self.train_split = train_split.copy()
            self.valid_split = valid_split.copy()
            self.mode_str = 'V2'  # new mode
        else:
            self.mode_str = 'V1'  # old mode
            self.data = data
            self.train_split = train_split.copy()
            self.valid_split = valid_split.copy()
            if check:
                intersection = set(train_split).intersection(set(valid_split))
                assert len(
                    intersection
                ) == 0, 'the splitted train and validation sets should have no intersection'
        self.length = len(self.train_split)

    def __repr__(self):
        return (
            '{name}(name={datasetname}, train={tr_L}, valid={val_L}, version={ver})'
            .format(
                name=self.__class__.__name__,
                datasetname=self.datasetname,
                tr_L=len(self.train_split),
                val_L=len(self.valid_split),
                ver=self.mode_str))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index >= 0 and index < self.length, 'invalid index = {:}'.format(
            index)
        train_index = self.train_split[index]
        valid_index = random.choice(self.valid_split)
        if self.mode_str == 'V1':
            train_image, train_label = self.data[train_index]
            valid_image, valid_label = self.data[valid_index]
        elif self.mode_str == 'V2':
            train_image, train_label = self.train_data[train_index]
            valid_image, valid_label = self.valid_data[valid_index]
        else:
            raise ValueError('invalid mode : {:}'.format(self.mode_str))
        return train_image, train_label, valid_image, valid_label


Dataset2Class = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet-1k-s': 1000,
    'imagenet-1k': 1000,
    'ImageNet16': 1000,
    'ImageNet16-150': 150,
    'ImageNet16-120': 120,
    'ImageNet16-200': 200
}


class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return ('{name}(length={length})'.format(
            name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


imagenet_pca = {
    'eigval':
    np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec':
    np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):

    def __init__(self,
                 alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3, )
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3, ))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def get_datasets(name, root, cutout):
    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('imagenet-1k'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('ImageNet16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
    else:
        raise TypeError('Unknow dataset : {:}'.format(name))

    # Data Argumentation
    if name in ['cifar10', 'cifar100']:
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name.startswith('ImageNet16'):
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        xshape = (1, 3, 16, 16)
    elif name == 'tiered':
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(80, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose([
            transforms.CenterCrop(80),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        xshape = (1, 3, 32, 32)
    elif name.startswith('imagenet-1k'):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if name == 'imagenet-1k':
            xlists = [transforms.RandomResizedCrop(224)]
            xlists.append(
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2))
            xlists.append(Lighting(0.1))
        elif name == 'imagenet-1k-s':
            xlists = [transforms.RandomResizedCrop(224, scale=(0.2, 1.0))]
        else:
            raise ValueError('invalid name : {:}'.format(name))
        xlists.append(transforms.RandomHorizontalFlip(p=0.5))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), normalize
        ])
        xshape = (1, 3, 224, 224)
    else:
        raise TypeError('Unknow dataset : {:}'.format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith('imagenet-1k'):
        train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
        test_data = dset.ImageFolder(osp.join(root, 'val'), test_transform)
        assert len(train_data) == 1281167 and len(
            test_data
        ) == 50000, 'invalid number of images : {:} & {:} vs {:} & {:}'.format(
            len(train_data), len(test_data), 1281167, 50000)
    elif name == 'ImageNet16':
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == 'ImageNet16-150':
        train_data = ImageNet16(root, True, train_transform, 150)
        test_data = ImageNet16(root, False, test_transform, 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == 'ImageNet16-200':
        train_data = ImageNet16(root, True, train_transform, 200)
        test_data = ImageNet16(root, False, test_transform, 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else:
        raise TypeError('Unknow dataset : {:}'.format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_nas_search_loaders(train_data, valid_data, dataset, config_root,
                           batch_size, workers):
    if isinstance(batch_size, (list, tuple)):
        batch, test_batch = batch_size
    else:
        batch, test_batch = batch_size, batch_size
    if dataset == 'cifar10':
        # split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
        cifar_split = load_config('{:}/cifar-split.txt'.format(config_root),
                                  None, None)
        # search over the proposed training and validation set
        train_split, valid_split = cifar_split.train, cifar_split.valid
        # they are two disjoint groups in the original CIFAR-10 training set
        # To split data
        xvalid_data = deepcopy(train_data)
        if hasattr(xvalid_data, 'transforms'):  # to avoid a print issue
            xvalid_data.transforms = valid_data.transform
        xvalid_data.transform = deepcopy(valid_data.transform)
        search_data = SearchDataset(dataset, train_data, train_split,
                                    valid_split)
        # data loader
        search_loader = torch.utils.data.DataLoader(
            search_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
            num_workers=workers,
            pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            xvalid_data,
            batch_size=test_batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
            num_workers=workers,
            pin_memory=True)
    elif dataset == 'cifar100':
        cifar100_test_split = load_config(
            '{:}/cifar100-test-split.txt'.format(config_root), None, None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data)
        search_valid_data.transform = train_data.transform
        search_data = SearchDataset(dataset,
                                    [search_train_data, search_valid_data],
                                    list(range(len(search_train_data))),
                                    cifar100_test_split.xvalid)
        search_loader = torch.utils.data.DataLoader(
            search_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=test_batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                cifar100_test_split.xvalid),
            num_workers=workers,
            pin_memory=True)
    elif dataset == 'ImageNet16-120':
        imagenet_test_split = load_config(
            '{:}/imagenet-16-120-test-split.txt'.format(config_root), None,
            None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data)
        search_valid_data.transform = train_data.transform
        search_data = SearchDataset(dataset,
                                    [search_train_data, search_valid_data],
                                    list(range(len(search_train_data))),
                                    imagenet_test_split.xvalid)
        search_loader = torch.utils.data.DataLoader(
            search_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=test_batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                imagenet_test_split.xvalid),
            num_workers=workers,
            pin_memory=True)
    else:
        raise ValueError('invalid dataset : {:}'.format(dataset))
    return search_loader, train_loader, valid_loader
