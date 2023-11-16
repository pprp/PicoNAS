import argparse
import collections
import copy

import numpy as np
from nasbench import api
from utils import get_bit_data_darts, padding_zeros_darts


class ArchDarts:
    def __init__(self, arch):
        self.arch = arch

    @classmethod
    def random_arch(cls):
        # output a uniformly random architecture spec
        # from the DARTS repository
        # https://github.com/quark0/darts
        NUM_VERTICES = 4
        OPS = [
            'none',
            'sep_conv_3x3',
            'dil_conv_3x3',
            'sep_conv_5x5',
            'dil_conv_5x5',
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect',
        ]
        normal = []
        reduction = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(1, len(OPS)), NUM_VERTICES)

            # input nodes for conv
            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            # input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

            normal.extend([(nodes_in_normal[0], ops[0]),
                          (nodes_in_normal[1], ops[1])])
            reduction.extend(
                [(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])]
            )
        return (normal, reduction)


class DataSetDarts:
    def __init__(self, dataset_num=int(1e6), dataset=None):
        self.dataset = 'darts'
        self.INPUT_1 = 'c_k-2'  # num 0
        self.INPUT_2 = 'c_k-1'  # num 1
        self.BASIC_MATRIX = [
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        # a mapping between genotype and op_list
        self.mapping_intermediate_node_ops = [
            {'input': 1},
            {'input': 2, 0: 5},
            {'input': 3, 0: 6, 1: 8},
            {'input': 4, 0: 7, 1: 9, 2: 10},
        ]
        self.op_integer = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: -1}
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = self.generate_random_dataset(dataset_num)
        print('Generate DARTS dataset, the size is :{}'.format(dataset_num))

    def generate_random_dataset(self, num):
        """
        create a dataset of randomly sampled architectures where may exist duplicates
        """
        data = []
        while len(data) < num:
            archtuple = ArchDarts.random_arch()
            data.append(archtuple)
        return data

    def get_ops(self, cell_tuple):
        all_ops = []
        mapping = self.mapping_intermediate_node_ops
        for c_k in range(2):
            # assign op list
            # initial ops are all zeros, i.e. all types are None
            ops = np.zeros(12, dtype='int8')
            # 'input' -2, 'output' -3
            input_output_integer = {'input': -2, 'output': -3}
            ops[0], ops[-1] = (
                input_output_integer['input'],
                input_output_integer['output'],
            )
            for position, op in enumerate(cell_tuple):
                intermediate_node = position // 2
                prev_node = op[0]
                if prev_node == 0 or prev_node == 1:
                    # is it 'input' or to ignore
                    if prev_node == c_k:
                        prev_node = 'input'
                    else:
                        prev_node = 'input_ignore'
                else:
                    # if is intermediate node, the number should minus 2
                    prev_node -= 2

                # determine the position in the ops
                if not prev_node == 'input_ignore':
                    ops_position = mapping[intermediate_node][prev_node]
                    op_type = op[1]
                    ops[ops_position] = op_type
            all_ops.append(ops)
        return all_ops

    def delete_useless_nodes(self, cell_tuple):
        """
        This function would not change the op integers (1-6)
        The skip connection is 7, the none is 0
        """
        all_matrix, all_ops, new_all_ops = [], self.get_ops(cell_tuple), []

        BASICMATRIX_LENGTH = len(self.BASIC_MATRIX)
        for ops in all_ops:
            matrix = copy.deepcopy(self.BASIC_MATRIX)
            for i, op in enumerate(ops):
                if op == 7:  # skip connection
                    m, n = [], []
                    for m_index in range(BASICMATRIX_LENGTH):
                        ele = matrix[m_index][i]
                        if ele == 1:
                            # set element to 0
                            matrix[m_index][i] = 0
                            m.append(m_index)

                    for n_index in range(BASICMATRIX_LENGTH):
                        ele = matrix[i][n_index]
                        if ele == 1:
                            # set element to 0
                            matrix[i][n_index] = 0
                            n.append(n_index)

                    for m_index in m:
                        for n_index in n:
                            matrix[m_index][n_index] = 1

                elif op == 0:  # none op type
                    for m_index in range(BASICMATRIX_LENGTH):
                        matrix[m_index][i] = 0
                    for n_index in range(BASICMATRIX_LENGTH):
                        matrix[i][n_index] = 0

            # start pruning
            model_spec = api.ModelSpec(matrix=matrix, ops=list(ops))
            all_matrix.append(model_spec.matrix)
            new_all_ops.append(model_spec.ops)
        return all_matrix, new_all_ops

    def transfer_ops(self, ops):
        """
        op_dict = {
                0: 'none',
                1: 'sep_conv_5x5',
                2: 'dil_conv_5x5',
                3: 'sep_conv_3x3',
                4: 'dil_conv_3x3',
                5: 'max_pool_3x3',
                6: 'avg_pool_3x3',
                7: 'skip_connect'
            }
        input darts ops, first delete the input and output, then change 1,2->-3; 3,4->2; 5,6->3
        -3 represents the type of operation that did not occur in the source domain
        :param ops: len=2
        """
        trans_ops = []
        for op in ops:
            trans_op = copy.deepcopy(op)
            trans_op = trans_op[1:-1]
            for index, op_value in enumerate(trans_op):
                if op_value == 1 or op_value == 2:
                    trans_op[index] = -3
                elif op_value == 3 or op_value == 4:
                    trans_op[index] = 2
                elif op_value == 5 or op_value == 6:
                    trans_op[index] = 3
                elif op_value == 0:
                    trans_op[index] = 0
                else:
                    raise ValueError('ops value should be from 0 to 6.')
            trans_ops.append(trans_op)

        return trans_ops

    def get_architecture_info(self, transfer_ops=True):
        DartsArchitectureSet = collections.OrderedDict()
        for index, tuple_arch in enumerate(self.dataset):
            norm_matrixes, norm_ops = self.delete_useless_nodes(tuple_arch[0])
            reduc_matrixes, reduc_ops = self.delete_useless_nodes(
                tuple_arch[1])

            padding_norm_matrixes, padding_norm_ops = padding_zeros_darts(
                norm_matrixes, norm_ops
            )
            padding_reduc_matrixes, padding_reduc_ops = padding_zeros_darts(
                reduc_matrixes, reduc_ops
            )

            if transfer_ops:
                padding_norm_ops = self.transfer_ops(padding_norm_ops)
                padding_reduc_ops = self.transfer_ops(padding_reduc_ops)

            tuple_arch_info = {
                'padding_norm_matrixes': padding_norm_matrixes,
                'padding_norm_ops': padding_norm_ops,
                'padding_reduc_matrixes': padding_reduc_matrixes,
                'padding_reduc_ops': padding_reduc_ops,
            }
            DartsArchitectureSet[index] = tuple_arch_info
        return DartsArchitectureSet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NB201 test')
    parser.add_argument(
        '--integers2one_hot',
        type=bool,
        default=True,
        help='whether to transform integers -> one_hot',
    )
    args = parser.parse_args()

    Darts = DataSetDarts(100)
    DartsSet = Darts.get_architecture_info(transfer_ops=True)
    X = get_bit_data_darts(DartsSet, integers2one_hot=args.integers2one_hot)
    print()
