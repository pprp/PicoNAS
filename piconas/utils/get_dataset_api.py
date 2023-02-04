import json
import os
import pickle
import random
from collections import namedtuple
from pathlib import Path

import torch

from piconas.models.nds import AnyNet, NetworkCIFAR, NetworkImageNet

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
"""
This file loads any dataset files or api's needed by the Trainer or ZeroCostPredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""

TASK_NAMES = [
    'autoencoder',
    'class_object',
    'class_scene',
    'normal',
    'jigsaw',
    'room_layout',
    'segmentsemantic',
]


def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent.parent


def get_transbench101_api(dataset):
    """
    Load the TransNAS-Bench-101 data
    """
    if dataset not in TASK_NAMES:
        return None

    datafile_path = os.path.join(get_project_root(), 'data',
                                 'transnas-bench_v10141024.pth')
    assert os.path.exists(
        datafile_path
    ), f'Could not fine {datafile_path}. Please download transnas-bench_v10141024.pth\
 from https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101'

    from piconas.nas.search_spaces import TransNASBenchAPI

    api = TransNASBenchAPI(datafile_path)
    return {'api': api, 'task': dataset}


def get_nasbench101_api(dataset=None):
    import piconas.nas.utils.nb101_api as api

    nb101_datapath = os.path.join(get_project_root(), 'data',
                                  'nasbench_only108.pkl')
    assert os.path.exists(
        nb101_datapath
    ), f'Could not find {nb101_datapath}. Please download nasbench_only108.pk \
from https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa'

    nb101_data = api.NASBench(nb101_datapath)
    return {'api': api, 'nb101_data': nb101_data}


def get_nasbench201_api(dataset):
    """
    Load the NAS-Bench-201 data
    """
    datafiles = {
        'cifar10': 'nb201_cifar10_full_training.pickle',
        'cifar100': 'nb201_cifar100_full_training.pickle',
        'ImageNet16-120': 'nb201_ImageNet16_full_training.pickle',
    }

    if dataset not in datafiles.keys():
        return None

    datafile_path = os.path.join(get_project_root(), 'data',
                                 datafiles[dataset])
    assert os.path.exists(
        datafile_path
    ), f'Could not find {datafile_path}. Please download {datafiles[dataset]} from \
https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa'

    with open(datafile_path, 'rb') as f:
        data = pickle.load(f)

    return {'nb201_data': data}


def get_nasbench301_api(dataset):
    if dataset != 'cifar10':
        return None
    # Load the nb301 performance and runtime models
    try:
        import nasbench301
    except ModuleNotFoundError:
        raise ModuleNotFoundError("No module named 'nasbench301'. \
            Please install nasbench301 from https://github.com/automl/nasbench301@no_gin"
                                  )

    # Paths to v1.0 model files and data file.
    download_path = os.path.join(get_project_root(), 'data', 'nb301_models')
    nb_models_path = os.path.join(download_path, 'nb_models_1.0')
    os.makedirs(download_path, exist_ok=True)

    nb301_model_path = os.path.join(nb_models_path, 'xgb_v1.0')
    nb301_runtime_path = os.path.join(nb_models_path, 'lgb_runtime_v1.0')

    if not all(
            os.path.exists(model)
            for model in [nb301_model_path, nb301_runtime_path]):
        nasbench301.download_models(
            version='1.0', delete_zip=True, download_dir=download_path)

    models_not_found_msg = 'Please download v1.0 models from \
https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510'

    # Verify the model and data files exist
    assert os.path.exists(
        nb_models_path
    ), f'Could not find {nb_models_path}. {models_not_found_msg}'
    assert os.path.exists(
        nb301_model_path
    ), f'Could not find {nb301_model_path}. {models_not_found_msg}'
    assert os.path.exists(
        nb301_runtime_path
    ), f'Could not find {nb301_runtime_path}. {models_not_found_msg}'

    performance_model = nasbench301.load_ensemble(nb301_model_path)
    runtime_model = nasbench301.load_ensemble(nb301_runtime_path)

    nb301_model = [performance_model, runtime_model]

    return {
        'nb301_model': nb301_model,
    }


class ReturnFeatureLayer(torch.nn.Module):

    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod(x), x


def return_feature_layer(network, prefix=''):
    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + '\t')


class NDS:

    def __init__(self, searchspace):
        self.searchspace = searchspace
        data = json.load(open(f'nds_data/{searchspace}.json', 'r'))
        try:
            data = data['top'] + data['mid']
        except Exception as e:
            pass
        self.data = data

    def __iter__(self):
        for unique_hash in range(len(self)):
            network = self.get_network(unique_hash)
            yield unique_hash, network

    def get_network_config(self, uid):
        return self.data[uid]['net']

    def get_network_optim_config(self, uid):
        return self.data[uid]['optim']

    def get_network(self, uid):
        netinfo = self.data[uid]
        config = netinfo['net']
        # print(config)
        if 'genotype' in config:
            # print('geno')
            gen = config['genotype']
            genotype = Genotype(
                normal=gen['normal'],
                normal_concat=gen['normal_concat'],
                reduce=gen['reduce'],
                reduce_concat=gen['reduce_concat'])
            if '_in' in self.searchspace:
                network = NetworkImageNet(config['width'], 1, config['depth'],
                                          config['aux'], genotype)
            else:
                network = NetworkCIFAR(config['width'], 1, config['depth'],
                                       config['aux'], genotype)
            network.drop_path_prob = 0.
            # print(config)
            # print('genotype')
            L = config['depth']
        else:
            if 'bot_muls' in config and 'bms' not in config:
                config['bms'] = config['bot_muls']
                del config['bot_muls']
            if 'num_gs' in config and 'gws' not in config:
                config['gws'] = config['num_gs']
                del config['num_gs']
            config['nc'] = 1
            config['se_r'] = None
            config['stem_w'] = 12
            L = sum(config['ds'])
            if 'ResN' in self.searchspace:
                config['stem_type'] = 'res_stem_in'
            else:
                config['stem_type'] = 'simple_stem_in'
            # "res_stem_cifar": ResStemCifar,
            # "res_stem_in": ResStemIN,
            # "simple_stem_in": SimpleStemIN,
            if config['block_type'] == 'double_plain_block':
                config['block_type'] = 'vanilla_block'
            network = AnyNet(**config)
        return_feature_layer(network)
        return network

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.data)

    def random_arch(self):
        return random.randint(0, len(self.data) - 1)

    def get_final_accuracy(self, uid, acc_type, trainval):
        return 100. - self.data[uid]['test_ep_top1'][-1]


def get_dataset_api(search_space=None, dataset=None):

    if search_space == 'nasbench101':
        return get_nasbench101_api(dataset=dataset)

    elif search_space == 'nasbench201':
        return get_nasbench201_api(dataset=dataset)

    elif search_space == 'nasbench301':
        return get_nasbench301_api(dataset=dataset)

    elif search_space in [
            'transbench101',
            'transbench101_micro',
            'transbench101_macro',
    ]:
        return get_transbench101_api(dataset=dataset)

    elif search_space == 'nds_resnet':
        return NDS('ResNet')
    elif search_space == 'nds_amoeba':
        return NDS('Amoeba')
    elif search_space == 'nds_amoeba_in':
        return NDS('Amoeba_in')
    elif search_space == 'nds_darts_in':
        return NDS('DARTS_in')
    elif search_space == 'nds_darts':
        return NDS('DARTS')
    elif search_space == 'nds_darts_fix-w-d':
        return NDS('DARTS_fix-w-d')
    elif search_space == 'nds_darts_lr-wd':
        return NDS('DARTS_lr-wd')
    elif search_space == 'nds_enas':
        return NDS('ENAS')
    elif search_space == 'nds_enas_in':
        return NDS('ENAS_in')
    elif search_space == 'nds_enas_fix-w-d':
        return NDS('ENAS_fix-w-d')
    elif search_space == 'nds_pnas':
        return NDS('PNAS')
    elif search_space == 'nds_pnas_fix-w-d':
        return NDS('PNAS_fix-w-d')
    elif search_space == 'nds_pnas_in':
        return NDS('PNAS_in')
    elif search_space == 'nds_nasnet':
        return NDS('NASNet')
    elif search_space == 'nds_nasnet_in':
        return NDS('NASNet_in')
    elif search_space == 'nds_resnext-a':
        return NDS('ResNeXt-A')
    elif search_space == 'nds_resnext-a_in':
        return NDS('ResNeXt-A_in')
    elif search_space == 'nds_resnext-b':
        return NDS('ResNeXt-B')
    elif search_space == 'nds_resnext-b_in':
        return NDS('ResNeXt-B_in')
    elif search_space == 'nds_vanilla':
        return NDS('Vanilla')
    elif search_space == 'nds_vanilla_lr-wd':
        return NDS('Vanilla_lr-wd')
    elif search_space == 'nds_vanilla_lr-wd_in':
        return NDS('Vanilla_lr-wd_in')
    else:
        raise NotImplementedError()


def get_zc_benchmark_api(search_space, dataset):

    datafile_path = os.path.join(get_project_root(), 'data',
                                 f'zc_{search_space}.json')
    with open(datafile_path) as f:
        data = json.load(f)

    return data[dataset]


def load_sampled_architectures(search_space, postfix=''):
    datafile_path = os.path.join(get_project_root(), 'data', 'archs',
                                 f'archs_{search_space}{postfix}.json')
    with open(datafile_path) as f:
        data = json.load(f)

    return data
