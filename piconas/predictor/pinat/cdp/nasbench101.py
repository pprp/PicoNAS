import argparse
import os
import pickle

import numpy as np
from nasbench import api

import piconas.predictor.pinat.cdp.utils as utils

NASBENCH_TFRECORD = os.path.join('path', 'nasbench_only108.tfrecord')

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'


class NB101:
    def __init__(self):
        self.dataset = 'nas-bench-101'

    def get_all_metrics(self):
        # use the len of index_list and the first index to distinguish different index_list
        save_path = os.path.join('path', 'tiny_nas_bench_101.pkl')
        if not os.path.isfile(save_path):
            nasbench = api.NASBench(NASBENCH_TFRECORD)
            important_metrics = {}
            for iter_num, unique_hash in enumerate(nasbench.hash_iterator()):
                fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(
                    unique_hash
                )
                final_training_time_list = []
                final_valid_accuracy_list = []
                for i in range(3):
                    # the 108 means the train epochs, and I have only downloaded the metrics in Epoch 108 (full train)
                    # the three iterations: three results of independent experiments recorded in the dataset
                    final_training_time_list.append(
                        computed_metrics[108][i]['final_training_time']
                    )
                    final_valid_accuracy_list.append(
                        computed_metrics[108][i]['final_validation_accuracy']
                    )
                # use the mean of three metrics
                final_training_time = np.mean(final_training_time_list)
                final_valid_accuracy = np.mean(final_valid_accuracy_list)

                # using the index to create dicts
                important_metrics[iter_num] = {}
                important_metrics[iter_num]['fixed_metrics'] = fixed_metrics
                important_metrics[iter_num]['final_training_time'] = final_training_time
                important_metrics[iter_num][
                    'final_valid_accuracy'
                ] = final_valid_accuracy

            if not os.path.isdir('pkl'):
                os.mkdir('pkl')

            with open(save_path, 'wb') as file:
                pickle.dump(important_metrics, file)
        else:
            with open(save_path, 'rb') as file:
                important_metrics = pickle.load(file)

        return important_metrics

    # transform the operations list to integers list
    # input: important_metrics
    # output: the metrics after padding
    def operations2integers(self, important_metrics):
        dict_oper2int = {NULL: 0, CONV1X1: 1, CONV3X3: 2, MAXPOOL3X3: 3}
        for i in important_metrics:
            fix_metrics = important_metrics[i]['fixed_metrics']
            module_operations = fix_metrics['module_operations']
            module_integers = np.array(
                [dict_oper2int[x] for x in module_operations[1:-1]]
            )
            # use [1: -1] to remove 'input' and 'output'
            important_metrics[i]['fixed_metrics']['module_integers'] = module_integers
        return important_metrics

    def get_main_data(self):
        metrics = self.get_all_metrics()
        # padding to 9*9
        metrics = utils.padding_zero_in_matrix(metrics)
        metrics = self.operations2integers(metrics)

        return metrics

    def load_data_101(self, integers2one_hot):
        metrics = self.get_main_data()
        X, y = utils.get_bit_data(metrics, integers2one_hot=integers2one_hot)
        return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NB101 test')
    parser.add_argument(
        '--integers2one_hot',
        type=bool,
        default=True,
        help='whether to transform integers -> one_hot',
    )
    args = parser.parse_args()

    dataset = NB101()
    metrics = dataset.load_data_101(args.integers2one_hot)
    print()
