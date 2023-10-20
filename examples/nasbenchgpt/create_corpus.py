import csv
import json

import numpy as np
from nas_201_api import NASBench201API as API
from tqdm import tqdm

api = API('/data2/dongpeijie/share/bench/NAS-Bench-201-v1_1-096897.pth', verbose=False)

search_space_infos = {
    'nas-bench-201-bak':
    """Each architecture consists of a
            predefined skeleton with a stack of the searched cell,
            where a cell is represented as a directed acyclic graph.
            Each edge is associated with an operation selected from
            a predifined set of operations. The operations are applied
            to the input nodes of the edge and the output of the edge
            is the result of the operation. The output of the cell is
            the concatenation of the outputs of all edges. The architecture
            is represented as a sequence of cells, where each cell is
            connected to the previous cell and the input node of the
            first cell is the input node of the architecture. The candidate
            operations are `nor_conv_3x3` representing a 3x3 convolutional
            layer, `avg_pool_3x3` representing a 3x3 average pooling layer,
            `nor_conv_1x1` representing a 1x1 convolutional layer,
            `skip_connect` representing a skip connection, and `none`
            representing no operation""",
    'nas-bench-201':
    'operations are nor_conv_3x3, avg_pool_3x3, nor_conv_1x1, skip_connect, or none.'
}

datasets_infos = {
    'cifar10-valid': 'training on the CIFAR-10 training set.',
    'cifar10': 'training on the CIFAR-10 training + validation set.',
    'cifar100': 'training on the CIFAR-100 training set.',
    'ImageNet16-120': 'training on the ImageNet16-120 training set.',
}

settings_infos = {
    'nas-bench-201-bak':
    """Batch size is 256, optimizer is SGD,
            initial learning rate is 0.1, ending learning rate is 0,
            weight decay is 0.0005, momentum is 0.9, and the number
            of epochs is 200. Random flip and crop are used for
            data augmentation.""",
    'nas-bench-201':
    'Batch size: 256, Optimizer: SGD, LR: 0.1-0, WD: 0.0005, Momentum: 0.9, Epochs: 200. Data aug: Rand. flip & crop.'
}


def convert_nb201_template():
    """ TEMPLATE 01: full version with 150M data """

    template = dict()
    template['type'] = 'text_only'
    template['instances'] = [] 

    for i in tqdm(range(len(api))):
        moving_dict = dict() 
        info = api.query_meta_info_by_index(i)
        arch_str = api[i]

        print(f'idx: {i} ')
        valacc = info.get_metrics('cifar10-valid', 'x-valid')['accuracy']

        # validacc_dict = {
        #     'instruction':
        #     f'search space: {search_space_infos["nas-bench-201"]}'
        #     f'dataset: {datasets_infos["cifar10-valid"]}'
        #     # f'setting: {settings_infos["nas-bench-201"]}'
        #     f'Based on the above information,'
        #     f'predict the valid accuracy of given architecture',
        #     'input':          f'arch is {arch_str}',
        #     'output':         f'valid accuracy is {valacc}'
        # }

        validacc_string = f"Search space is {search_space_infos['nas-bench-201']}. " \
                        f"Dataset is {datasets_infos['cifar10-valid']}. " \
                        f"Based on the above information, predict the valid accuracy of given architecture. " \
                        f"### User: the architecture string is {arch_str}." \
                        f"### Agent: the valid accuracy is {valacc}."

        trainacc = info.get_metrics('cifar10', 'train')['accuracy']

        # trainacc_dict = {
        #     'instruction':
        #     f'search space: {search_space_infos["nas-bench-201"]}'
        #     f'dataset: {datasets_infos["cifar10-valid"]}'
        #     # f'setting: {settings_infos["nas-bench-201"]}'
        #     f'Based on the above information,'
        #     f'Predict the train accuracy of given architecture',
        #     'input':
        #     f'arch is {arch_str}',
        #     'output':
        #     f'valid accuracy is {trainacc}'
        # }
        trainacc_string = f"Search space is {search_space_infos['nas-bench-201']}. " \
                f"Dataset is {datasets_infos['cifar10-valid']}. " \
                f"Based on the above information, predict the train accuracy of given architecture. " \
                f"### User: the architecture string is {arch_str}." \
                f"### Agent: the train accuracy is {trainacc}."

        valloss = info.get_metrics('cifar10-valid', 'x-valid')['loss']

        # validloss_dict = {
        #     'instruction':
        #     f'search space: {search_space_infos["nas-bench-201"]}'
        #     f'dataset: {datasets_infos["cifar10-valid"]}'
        #     # f'setting: {settings_infos["nas-bench-201"]}'
        #     f'Based on the above information,'
        #     f'predict the valid loss of given architecture',
        #     'input':
        #     f'arch is {arch_str}',
        #     'output':
        #     f'valid loss is {valloss}'
        # }
        validloss_string = f"Search space is {search_space_infos['nas-bench-201']}. " \
                 f"Dataset is {datasets_infos['cifar10-valid']}. " \
                 f"Based on the above information, predict the valid loss of given architecture. " \
                 f"### User: the architecture string is {arch_str}." \
                 f"### Agent: the valid loss is {valloss}."


        trainloss = info.get_metrics('cifar10-valid', 'train')['loss']

        # trainloss_dict = {
        #     'instruction':
        #     f'search space: {search_space_infos["nas-bench-201"]}'
        #     f'dataset: {datasets_infos["cifar10-valid"]}'
        #     # f'setting: {settings_infos["nas-bench-201"]}'
        #     f'Based on the above information,'
        #     f'Predict the train loss of given architecture',
        #     'input':
        #     f'arch is {arch_str}',
        #     'output':
        #     f'valid loss is {trainloss}'
        # }
        trainloss_string = f"Search space is {search_space_infos['nas-bench-201']}. " \
                 f"Dataset is {datasets_infos['cifar10-valid']}. " \
                 f"Based on the above information, predict the train loss of given architecture. " \
                 f"### User: the architecture string is {arch_str}." \
                 f"### Agent: the train loss is {trainloss}."


        flops = info.get_compute_costs('cifar10-valid')['flops']

        # flops_dict = {
        #     'instruction':
        #     f'search space: {search_space_infos["nas-bench-201"]}'
        #     f'dataset: {datasets_infos["cifar10-valid"]}'
        #     # f'setting: {settings_infos["nas-bench-201"]}'
        #     f'Based on the above information,'
        #     f'Predict the flops of given architecture',
        #     'input':
        #     f'arch is {arch_str}',
        #     'output':
        #     f'flops is {flops}'
        # }
        flops_string = f"Search space is {search_space_infos['nas-bench-201']}. " \
             f"Dataset is {datasets_infos['cifar10-valid']}. " \
             f"Based on the above information, predict the flops of given architecture. " \
             f"### User: the architecture string is {arch_str}." \
             f"### Agent: the flops is {flops}."


        params = info.get_compute_costs('cifar10-valid')['params']

        # params_dict = {
        #     'instruction':
        #     f'search space: {search_space_infos["nas-bench-201"]}'
        #     f'dataset: {datasets_infos["cifar10-valid"]}'
        #     # f'setting: {settings_infos["nas-bench-201"]}'
        #     f'Based on the above information,'
        #     f'Predict the params of given architecture',
        #     'input':
        #     f'arch is {arch_str}',
        #     'output':
        #     f'params is {params}'
        # }
        params_string = f"Search space is {search_space_infos['nas-bench-201']}. " \
              f"Dataset is {datasets_infos['cifar10-valid']}. " \
              f"Based on the above information, predict the params of given architecture. " \
              f"### User: the architecture string is {arch_str}." \
              f"### Agent: the params is {params}."


        latency = info.get_compute_costs('cifar10-valid')['latency']

        # latency_dict = {
        #     'instruction':
        #     f'search space: {search_space_infos["nas-bench-201"]}'
        #     f'dataset: {datasets_infos["cifar10-valid"]}'
        #     # f'setting: {settings_infos["nas-bench-201"]}'
        #     f'Based on the above information,'
        #     f'Predict the latency of given architecture',
        #     'input':
        #     f'arch is {arch_str}',
        #     'output':
        #     f'latency is {latency}'
        # }

        latency_string = f"Search space is {search_space_infos['nas-bench-201']}. " \
               f"Dataset is {datasets_infos['cifar10-valid']}. " \
               f"Based on the above information, predict the latency of given architecture. " \
               f"### User: the architecture string is {arch_str}." \
               f"### Agent: the latency is {latency}."


        moving_dict["text"] = validacc_string + " " + \
              trainacc_string + " " + \
                validloss_string + " " + \ 
        trainloss_string + " " + \ 
        flops_string + " " + \ 
        params_string + " " + \ 
        + latency_string
        template["instances"].append(moving_dict)
    # convert template to json file
    with open('data/nb201_template.json', 'w') as f:
        json.dump(template, f)

if __name__ == '__main__':
    convert_nb201_template()
