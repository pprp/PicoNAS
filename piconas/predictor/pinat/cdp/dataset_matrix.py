import piconas.predictor.pinat.cdp.nasbench101 as nasbench101
import piconas.predictor.pinat.cdp.nasbench201 as nasbench201 
import piconas.predictor.pinat.cdp.darts as darts 
import piconas.predictor.pinat.cdp.tiny_darts as tiny_darts
from torch.utils.data import Dataset
import numpy as np
import random
import pickle
from utils import get_matrix_data_darts
import os


class Dataset_Train(Dataset):
    MEAN = 0.900721
    STD = 0.059585

    MEAN101 = 0.902434
    STD101 = 0.058647

    MEAN201 = 0.853237
    STD201 = 0.065461

    def __init__(self, split_type, normal_layer, percentile, using_dataset='all'):
        '''
        :param split_type: int type
        0: normal_cell0
        1: normal_cell1
        2: reduction_cell0
        3: reduction_cell1

        :normal_rate: float type between (0, 1)
        '''

        self.normal_rate = normal_layer / (normal_layer + 2)

        # initial is none
        self.adjacency = []
        self.operations = []
        self.num_vertices = []
        self.accuracy = []

        dataset101 = nasbench101.NB101()
        dataset201 = nasbench201.NB201()
        Dataset_Metrics101 = dataset101.get_main_data()
        Dataset_Metrics201 = dataset201.get_main_data()

        if percentile:
            NB101_acc = [self.normalize(Dataset_Metrics101[i]['final_valid_accuracy'], '1') for i in Dataset_Metrics101]
            NB201_acc = [self.normalize(Dataset_Metrics201[i]['final_valid_accuracy'], '2') for i in Dataset_Metrics201]
            all_acc = NB101_acc + NB201_acc
            self.percentile = []
            # the max class number
            K = 5
            for i in range(1, K+1):
                i_percentile = []
                step = 100 / i
                for j in range(1, i+1):
                    i_percentile.append(np.percentile(all_acc, min(step*j, 100)))
                self.percentile.append(i_percentile)

        # dataset101 and dataset201
        dataset_num = 0
        if using_dataset == 'all':
            dataset_list = [Dataset_Metrics101, Dataset_Metrics201]
        elif using_dataset == '101':
            dataset_list = [Dataset_Metrics101]
        elif using_dataset == '201':
            dataset_list = [Dataset_Metrics201]
        else:
            raise NotImplementedError
        for DataSet in dataset_list:
            dataset_num += 1
            for index in DataSet:
                fixed_metrics = DataSet[index]['fixed_metrics']
                accuracy = self.normalize(DataSet[index]['final_valid_accuracy'], dataset=str(dataset_num))
                adjacency_matrix = fixed_metrics['module_adjacency']
                module_integers = [-1] + list(fixed_metrics['module_integers']) + [-2]

                ops_onehot = np.array([[i == k + 2 for i in range(6)] for k in module_integers], dtype=np.float32)
                num_vert = len(module_integers) - module_integers.count(0)

                # append
                self.adjacency.append(adjacency_matrix)
                self.operations.append(ops_onehot)
                self.num_vertices.append(num_vert)
                self.accuracy.append(accuracy)

        # random shuffle
        random.shuffle(self.adjacency, random=random.seed(1))
        random.shuffle(self.operations, random=random.seed(1))
        random.shuffle(self.num_vertices, random=random.seed(1))
        random.shuffle(self.accuracy, random=random.seed(1))

        # set split range
        all_dataset_len = len(self.accuracy)
        normal_len = int(np.floor(self.normal_rate * all_dataset_len))
        reduction_len = all_dataset_len - normal_len
        split_point = [0, normal_len // 2, normal_len, normal_len + reduction_len // 2, all_dataset_len]
        split_range_list = [list(range(split_point[0], split_point[1])),
                            list(range(split_point[1], split_point[2])),
                            list(range(split_point[2], split_point[3])),
                            list(range(split_point[3], split_point[4]))]

        self.sample_range = split_range_list[split_type]

    def __len__(self):
        return len(self.sample_range)

    def normalize(self, num, dataset):
        if dataset == '1':
            mean = self.MEAN101
            std = self.STD101
        elif dataset == '2':
            mean = self.MEAN201
            std = self.STD201
        else:
            raise ValueError()
        return (num - mean) / std

    # @classmethod
    # def denormalize(cls, num):
    #     return num * cls.STD + cls.MEAN

    def __getitem__(self, index):
        index = self.sample_range[index]
        val_acc = self.accuracy[index]

        result = {
            "num_vertices": self.num_vertices[index],
            "adjacency": self.adjacency[index],
            "operations": self.operations[index],
            "val_acc": val_acc
        }
        return result


class Dataset_Darts(Dataset):
    def __init__(self, dataset_num=None, dataset=None, dataset_type='normal', ns=False):
        # initial is none
        self.operations = []
        self.num_vertices = []

        # select dataset type
        if dataset_type == 'normal':
            # normal means Darts
            # original description
            if dataset_num == None:
                self.dataset = darts.DataSetDarts(dataset_num=1e6)
                # darts_save_path = 'path/darts_dataset.pkl'
                # with open(darts_save_path, 'wb') as file:
                #     pickle.dump(self.dataset, file)
            else:
                self.dataset = darts.DataSetDarts(dataset_num=dataset_num, dataset=dataset)
        elif dataset_type == 'tiny':
            if dataset_num == None:
                self.dataset = tiny_darts.DataSetTinyDarts(dataset_num=1e6, no_skip_and_none=ns)
            else:
                self.dataset = tiny_darts.DataSetTinyDarts(dataset_num=dataset_num, dataset=dataset, no_skip_and_none=ns)
        elif dataset_type == 'small_tiny':
            if dataset_num == None:
                self.dataset = tiny_darts.DataSetSmallTinyDarts(dataset_num=1e6)
            else:
                self.dataset = tiny_darts.DataSetSmallTinyDarts(dataset_num=dataset_num, dataset=dataset)
        '''
        save_path = 'path/arch_info.pkl'
        if os.path.exists(save_path):
            with open(save_path, 'rb') as file:
                info_dict = pickle.load(file)
            DartsSet, m, o = info_dict['DartsSet'], info_dict['m'], info_dict['o']
        else:
        '''
        
        DartsSet = self.dataset.get_architecture_info(transfer_ops=True)
        m, o = get_matrix_data_darts(DartsSet)
        # info_dict = {'DartsSet': DartsSet, 'm': m, 'o': o}
        # with open(save_path, 'wb') as file:
        #     pickle.dump(info_dict, file)
        self.adjacency = m

        for ops in o:
            operations_sub, num_vertices_sub = [], []
            for op in ops:
                op_integers = [-1] + list(op) + [-2]
                # 1 denotes for 5*5 conv
                op_onehot = np.array([[i == k + 2 for i in range(6)] for k in op_integers], dtype=np.float32)
                num_vert = len(op_integers) - op_integers.count(0)
                operations_sub.append(op_onehot)
                num_vertices_sub.append(num_vert)
            self.operations.append(operations_sub)
            self.num_vertices.append(num_vertices_sub)

        assert len(self.adjacency) == len(self.operations) == len(self.num_vertices)

    def __len__(self):
        return len(self.adjacency)

    def __getitem__(self, index):
        result = {
            "num_vertices": self.num_vertices[index],
            "adjacency": self.adjacency[index],
            "operations": self.operations[index]
        }

        return result


if __name__ == '__main__':
    import os
    import torch
    from cross_domain_predictor import get_target_train_dataloader

    filename = os.path.join('eval-DATASET-20210320-171343', 'darts_dataset.pth.tar')
    data = torch.load(filename)
    target_dataloader = get_target_train_dataloader(train_batch_size=100, dataset_num=len(data['dataset']),
                                                    dataset=data['dataset'])
    print()