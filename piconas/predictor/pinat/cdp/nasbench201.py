import pickle
import os
import collections
from nasbench import api
from nas_201_api import NASBench201API as API201
import copy
import numpy as np
import piconas.predictor.pinat.cdp.utils as utils
import argparse


class NB201():
    def __init__(self):
        # basic matrix for nas_bench 201
        self.BASIC_MATRIX = [[0, 1, 1, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0]]

        self.NULL = 'null'
        self.CONV1X1 = 'nor_conv_1x1'
        self.CONV3X3 = 'nor_conv_3x3'
        self.AP3X3 = 'avg_pool_3x3'

    def delete_useless_node(self, ops):
        # delete the skip connections nodes and the none nodes
        # output the pruned metrics
        # start to change matrix
        matrix = copy.deepcopy(self.BASIC_MATRIX)
        for i, op in enumerate(ops, start=1):
            m = []
            n = []

            if op == 'skip_connect':
                for m_index in range(8):
                    ele = matrix[m_index][i]
                    if ele == 1:
                        # set element to 0
                        matrix[m_index][i] = 0
                        m.append(m_index)

                for n_index in range(8):
                    ele = matrix[i][n_index]
                    if ele == 1:
                        # set element to 0
                        matrix[i][n_index] = 0
                        n.append(n_index)

                for m_index in m:
                    for n_index in n:
                        matrix[m_index][n_index] = 1

            elif op == 'none':
                for m_index in range(8):
                    matrix[m_index][i] = 0
                for n_index in range(8):
                    matrix[i][n_index] = 0

        ops_copy = copy.deepcopy(ops)
        ops_copy.insert(0, 'input')
        ops_copy.append('output')

        # start pruning
        model_spec = api.ModelSpec(matrix=matrix, ops=ops_copy)
        return model_spec.matrix, model_spec.ops

    def save_arch_str2op_list(self, save_arch_str):
        op_list = []
        save_arch_str_list = API201.str2lists(save_arch_str)
        op_list.append(save_arch_str_list[0][0][0])
        op_list.append(save_arch_str_list[1][0][0])
        op_list.append(save_arch_str_list[1][1][0])
        op_list.append(save_arch_str_list[2][0][0])
        op_list.append(save_arch_str_list[2][1][0])
        op_list.append(save_arch_str_list[2][2][0])
        return op_list

    def operation2integers(self, op_list):
        dict_oper2int = {self.NULL: 0, self.CONV1X1: 1, self.CONV3X3: 2, self.AP3X3: 3}
        module_integers = np.array([dict_oper2int[x] for x in op_list[1: -1]])
        return module_integers

    def get_all_metrics(self, ordered_dic, dataset):
        metrics = {}
        for index in range(len(ordered_dic)):
            final_valid_acc = ordered_dic[index][dataset]
            epoch12_time = ordered_dic[index]['cifar10_all_time']
            op_list = self.save_arch_str2op_list(ordered_dic[index]['arch_str'])
            pruned_matrix, pruned_op = self.delete_useless_node(op_list)
            if pruned_matrix is None:
                continue
            padding_matrix, padding_op = utils.padding_zeros(pruned_matrix, pruned_op)
            op_integers = self.operation2integers(padding_op)

            metrics[index] = {'final_training_time': epoch12_time, 'final_valid_accuracy': final_valid_acc / 100}
            metrics[index]['fixed_metrics'] = {'module_adjacency': padding_matrix, 'module_integers': op_integers,
                                               'trainable_parameters': -1}
        return metrics

    def get_main_data(self):
        # load data
        tidy_file = r'path/tidy_nas_bench_201.pkl'
        if not os.path.exists(tidy_file):
            nasbench201 = API201(r'path/NAS-Bench-201-v1_1-096897.pth')
            ordered_dic = collections.OrderedDict()
            for index in range(len(nasbench201.evaluated_indexes)):
                info = nasbench201.query_meta_info_by_index(index, '12')
                arch_str = info.arch_str
                cifar10_valid = info.get_metrics('cifar10-valid', 'x-valid')['accuracy']
                cifar10_all_time = info.get_metrics('cifar10-valid', 'x-valid')['all_time']

                info = nasbench201.query_meta_info_by_index(index, '200')
                cifar10 = info.get_metrics('cifar10', 'ori-test')['accuracy']
                cifar10_valid200 = info.get_metrics('cifar10-valid', 'x-valid')['accuracy']
                index_info = {'arch_str': arch_str, 'cifar10': cifar10, 'cifar10_valid': cifar10_valid,
                              'cifar10_all_time': cifar10_all_time, 'cifar10_valid200': cifar10_valid200}
                ordered_dic[index] = index_info

            with open(tidy_file, 'wb') as file:
                pickle.dump(ordered_dic, file)
        else:
            with open(tidy_file, 'rb') as file:
                ordered_dic = pickle.load(file)

        NB201Data = self.get_all_metrics(ordered_dic, 'cifar10_valid200')
        return NB201Data

    def load_data_201(self, integers2one_hot):
        NB201Data = self.get_main_data()
        X, y = utils.get_bit_data(NB201Data, integers2one_hot=integers2one_hot)

        return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NB201 test')
    parser.add_argument('--integers2one_hot', type=bool, default=True, help='whether to transform integers -> one_hot')
    args = parser.parse_args()

    dataset = NB201()
    X, y = dataset.load_data_201(args.integers2one_hot)
    print('load {} pairs of data from nas-bench-201.'.format(len(X)))