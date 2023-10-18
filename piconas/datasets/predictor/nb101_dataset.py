import json
from copy import deepcopy

import dgl
import h5py
import numpy as np
import pandas as pd
import torch
from nasbench import api as NB1API
from scipy import sparse as sp
from torch.utils.data import Dataset

BASE_PATH = '/data2/dongpeijie/share/bench/predictor_embeddings/embedding_datasets/'


def laplacian_positional_encoding(adj, pos_enc_dim, number_nodes=7):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    lap_pos_enc = []
    num = 0
    for i in range(len(adj)):
        adj_matrix = adj[i] + adj[i].T
        degree = torch.tensor(np.sum(adj_matrix, axis=1))
        degree = sp.diags(
            dgl.backend.asnumpy(degree).clip(1)**-0.5, dtype=float)
        L = sp.eye(number_nodes) - degree * adj_matrix * degree
        try:
            EigVal, EigVec = sp.linalg.eigs(
                L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
        except:
            num += 1
            print('')
            print(adj_matrix)
            print(i, num)
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
        # lap_pos_enc.append(torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1].real).float())
        lap_pos_enc.append(EigVec[:, 1:pos_enc_dim + 1].real)

    #     if (i+1) % 10000 == 0:
    #         print(i)
    # np.save('pos_enc.npy', np.array(lap_pos_enc))
    # import sys
    # sys.exit()

    # Laplacian
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    # N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    # L = sp.eye(g.number_of_nodes()) - N * A * N
    # # Eigenvectors with scipy
    # #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    # return g
    return lap_pos_enc


def generate_lapla_matrix(adjacency, note):
    normalized_lapla_matrix = []
    unnormalized_lapla_matrix = []

    for index in range(len(adjacency)):
        # normalized lapla_matrix
        adj_matrix = adjacency[index]
        degree = torch.tensor(np.sum(adj_matrix, axis=1))
        degree = sp.diags(
            dgl.backend.asnumpy(degree).clip(1)**-0.5, dtype=float)
        normalized_lapla = np.array(
            sp.eye(adj_matrix.shape[0]) - degree * adj_matrix * degree)
        normalized_lapla_matrix.append(normalized_lapla)

        # un-normalized lapla_matrix
        adj_matrix = adjacency[index]
        degree = np.diag(np.sum(adj_matrix, axis=1))
        unnormalized_lapla = degree - adj_matrix
        unnormalized_lapla_matrix.append(unnormalized_lapla)

        if index % 100 == 0:
            print(note, index)

    np.save('%s_lapla_matrix.npy' % note, np.array(unnormalized_lapla_matrix))
    np.save('%s_lapla_nor_matrix.npy' % note,
            np.array(normalized_lapla_matrix))


class Nb101DatasetPINAT(Dataset):
    """
        Generate nasbench101 data stream for PINAT.
    """

    def __init__(self,
                 split=None,
                 debug=False,
                 candidate_ops=5,
                 pos_enc_dim=2,
                 data_type='train'):
        self.hash2id = dict()
        self.nb1_api = NB1API.NASBench(BASE_PATH +
                                       'nasbench_only108_caterec.tfrecord')
        self.hash_iterator_list = list(self.nb1_api.hash_iterator())
        with h5py.File(
                '/data2/dongpeijie/share/bench/pinat_bench_files/nasbench101/nasbench.hdf5',
                mode='r') as f:
            for i, h in enumerate(f['hash'][()]):
                self.hash2id[h.decode()] = i
            self.num_vertices = f['num_vertices'][()]
            self.trainable_parameters = f['trainable_parameters'][()]
            self.adjacency = f['adjacency'][()]
            self.operations = f['operations'][()]
            self.metrics = f['metrics'][()]
        self.random_state = np.random.RandomState(0)
        if split is not None and split != 'all':
            self.sample_range = np.load(
                '/data2/dongpeijie/share/bench/pinat_bench_files/nasbench101/train_samples.npz'
            )[str(split)]
        else:
            self.sample_range = list(range(len(self.hash2id)))
        self.debug = debug
        self.seed = 0
        self.candidate_ops = candidate_ops
        self.lapla = np.load(
            '/data2/dongpeijie/share/bench/pinat_bench_files/nasbench101/lapla_matrix.npy'
        )
        self.lapla_nor = np.load(
            '/data2/dongpeijie/share/bench/pinat_bench_files/nasbench101/lapla_nor_matrix.npy'
        )
        # load from '/data2/dongpeijie/share/bench/predictor_embeddings/embedding_datasets/zc_nasbench101.json'
        self.zc_score = json.load(
            open(
                '/data2/dongpeijie/share/bench/predictor_embeddings/embedding_datasets/zc_nasbench101.json',
                'r'))['cifar10']
        self.zcp_nb101 = json.load(
            open(BASE_PATH + 'zc_nasbench101_full.json', 'r'))

        self.normalize_and_process_zcp(True, True)
        self.data_type = data_type

        # all
        self.val_mean_mean, self.val_mean_std = 0.908192, 0.023961  # all val mean/std (from SemiNAS)
        self.test_mean_mean, self.test_mean_std = 0.8967984, 0.05799569  # all test mean/std

        self.max_edge_num = 9
        self.zcps = [
            'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov',
            'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen'
        ]

        self.nb1_op_idx = {
            'input': 0,
            'output': 1,
            'conv3x3-bn-relu': 2,
            'conv1x1-bn-relu': 3,
            'maxpool3x3': 4
        }
        # # get max_edge_num
        # max_edge_num = 0
        # for k, adjacency in enumerate(self.adjacency):
        #     edge_index = []
        #     for i in range(adjacency.shape[0]):
        #         idx_list = np.where(adjacency[i])[0].tolist()
        #         for j in idx_list:
        #             edge_index.append([i, j])
        #     if np.sum(edge_index) == 0:
        #         edge_index = []
        #         for i in range(adjacency.shape[0]):
        #             for j in range(adjacency.shape[0] - 1, i, -1):
        #                 edge_index.append([i, j])
        #     if len(edge_index) > max_edge_num:
        #         max_edge_num = len(edge_index)
        #     if k % 1000 == 0:
        #         print(k, max_edge_num)

    def __len__(self):
        return len(self.sample_range)

    def min_max_scaling(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def log_transform(self, data):
        return np.log1p(data)

    def standard_scaling(self, data):
        return (data - np.mean(data)) / np.std(data)

    def normalize_and_process_zcp(self, normalize_zcp, log_synflow=True):
        if normalize_zcp:
            print('Normalizing ZCP dict')
            try:
                self.norm_zcp = pd.DataFrame({
                    k0: {
                        k1: v1['score']
                        for k1, v1 in v0.items() if v1.__class__() == {}
                    }
                    for k0, v0 in self.zcp_nb101['cifar10'].items()
                }).T
            except KeyError:
                import pdb
                pdb.set_trace()

            # Add normalization code here
            self.norm_zcp['epe_nas'] = self.min_max_scaling(
                self.norm_zcp['epe_nas'])
            self.norm_zcp['fisher'] = self.min_max_scaling(
                self.log_transform(self.norm_zcp['fisher']))
            self.norm_zcp['flops'] = self.min_max_scaling(
                self.log_transform(self.norm_zcp['flops']))
            self.norm_zcp['grad_norm'] = self.min_max_scaling(
                self.log_transform(self.norm_zcp['grad_norm']))
            self.norm_zcp['grasp'] = self.standard_scaling(
                self.norm_zcp['grasp'])
            self.norm_zcp['jacov'] = self.min_max_scaling(
                self.norm_zcp['jacov'])
            self.norm_zcp['l2_norm'] = self.min_max_scaling(
                self.norm_zcp['l2_norm'])
            self.norm_zcp['nwot'] = self.min_max_scaling(self.norm_zcp['nwot'])
            self.norm_zcp['params'] = self.min_max_scaling(
                self.log_transform(self.norm_zcp['params']))
            self.norm_zcp['plain'] = self.min_max_scaling(
                self.norm_zcp['plain'])
            self.norm_zcp['snip'] = self.min_max_scaling(
                self.log_transform(self.norm_zcp['snip']))

            if log_synflow:
                self.norm_zcp['synflow'] = self.min_max_scaling(
                    self.log_transform(self.norm_zcp['synflow']))
            else:
                self.norm_zcp['synflow'] = self.min_max_scaling(
                    self.norm_zcp['synflow'])

            self.norm_zcp['zen'] = self.min_max_scaling(self.norm_zcp['zen'])
            self.norm_zcp['val_accuracy'] = self.min_max_scaling(
                self.norm_zcp['val_accuracy'])

            self.zcp_nb101 = {'cifar10': self.norm_zcp.T.to_dict()}

    def _check(self, item):
        n = item['num_vertices']
        ops = item['operations']
        adjacency = item['adjacency']
        mask = item['mask']
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0

    def mean_acc(self):
        return np.mean(self.metrics[:, -1, self.seed, -1, 2])

    def std_acc(self):
        return np.std(self.metrics[:, -1, self.seed, -1, 2])

    def normalize(self, num):
        if self.data_type == 'train':
            return (num - self.val_mean_mean) / self.val_mean_std
        elif self.data_type == 'test':
            return (num - self.test_mean_mean) / self.test_mean_std
        else:
            raise ValueError('Wrong data_type!')

    def denormalize(self, num):
        if self.data_type == 'train':
            return num * self.val_mean_std + self.val_mean_mean
        elif self.data_type == 'test':
            return num * self.test_mean_std + self.test_mean_mean
        else:
            raise ValueError('Wrong data_type!')

    def resample_acc(self, index, split='val'):
        # when val_acc or test_acc are out of range
        assert split in ['val', 'test']
        split = 2 if split == 'val' else 3
        for seed in range(3):
            acc = self.metrics[index, -1, seed, -1, split]
            if not self._is_acc_blow(acc):
                return acc
        if self.debug:
            print(index, self.metrics[index, -1, :, -1])
            raise ValueError
        return np.array(self.val_mean_mean)

    def _is_acc_blow(self, acc):
        return acc < 0.2

    def _rand_flip(self, batch_pos):
        batch_lap_pos_enc = torch.from_numpy(batch_pos)
        sign_flip = torch.rand(batch_lap_pos_enc.size(1))
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        return batch_lap_pos_enc

    def _check_validity(self, data_list, threshold=0.3):
        _data_list = deepcopy(data_list).tolist()
        del_idx = []
        for _idx, _data in enumerate(_data_list):
            if _data < threshold:
                del_idx.append(_idx)
        for _idx in del_idx[::-1]:
            _data_list.pop(_idx)
        if len(_data_list) == 0:
            return data_list
        else:
            return np.array(_data_list)

    def __getitem__(self, index):
        index = self.sample_range[index]
        val_acc = self.metrics[index, -1, :, -1, 2]
        test_acc = self.metrics[index, -1, :, -1, 3]
        val_acc = val_acc[0]

        test_acc = self._check_validity(test_acc)
        test_acc = np.mean(test_acc)
        if self._is_acc_blow(val_acc):
            val_acc = self.resample_acc(index, 'val')
        n = self.num_vertices[index]
        ops_onehot = np.array([[i == k + 2 for i in range(self.candidate_ops)]
                               for k in self.operations[index]],
                              dtype=np.float32)
        if n < 7:
            ops_onehot[n:] = 0.
        features = np.expand_dims(
            np.array([i for i in range(self.candidate_ops)]), axis=0)
        features = np.tile(features, (len(ops_onehot), 1))
        features = ops_onehot * features
        features = np.sum(features, axis=-1)
        adjacency = self.adjacency[index]
        lapla = self.lapla[index]
        lapla_nor = self.lapla_nor[index].astype(np.float32)

        # links
        edge_index = []
        for i in range(adjacency.shape[0]):
            idx_list = np.where(adjacency[i])[0].tolist()
            for j in idx_list:
                edge_index.append([i, j])
        if np.sum(edge_index) == 0:
            edge_index = []
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[0] - 1, i, -1):
                    edge_index.append([i, j])
        edge_num = len(edge_index)
        pad_num = self.max_edge_num - edge_num
        if pad_num > 0:
            edge_index = np.pad(
                np.array(edge_index), ((0, pad_num), (0, 0)),
                'constant',
                constant_values=(0, 0))
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.transpose(1, 0)

        # load normalized zc
        hash = self.hash_iterator_list[index]
        metrics = self.nb1_api.get_metrics_from_hash(hash)
        mat_adj = np.asarray(metrics[0]['module_adjacency']).flatten().tolist()
        mat_op = [self.nb1_op_idx[x] for x in metrics[0]['module_operations']]
        mat = mat_adj + mat_op
        mat_repr = str(tuple(mat))
        zcp = [self.zcp_nb101['cifar10'][mat_repr][nn] for nn in self.zcps]

        result = {
            'num_vertices': 7,
            'edge_num': edge_num,
            'adjacency': adjacency,
            'lapla': lapla,
            'lapla_nor': lapla_nor,
            'operations': ops_onehot,
            'features': torch.from_numpy(features).long(),
            'mask': np.array([i < n for i in range(7)], dtype=np.float32),
            'val_acc': float(self.normalize(val_acc)),
            'test_acc': float(self.normalize(test_acc)),
            'edge_index_list': edge_index,
            'zcp': zcp,
        }
        if self.debug:
            self._check(result)
        return result
