import argparse
import math
import os
import pickle
import time

# from dataset_matrix import Dataset_Train, Dataset_Darts
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import AverageMeterGroup, convert_to_genotype, get_logger, to_cuda


class DANN(nn.Module):

    def __init__(self, input_size=128):
        super(DANN, self).__init__()
        self.input_size = input_size
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1',
                                          nn.Linear(self.input_size, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_feature):
        # input_feature = input_feature.view(-1, self.input_size)
        # input_feature = input_feature.view(-1)
        domain_output = self.domain_classifier(input_feature)

        return domain_output


def guassian_kernel(source,
                    target,
                    kernel_mul=2.0,
                    kernel_num=5,
                    fix_sigma=None):
    '''
    Learning Transferable Features with Deep Adaptation Networks
    https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DAN/mmd.py
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul**(kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp)
        for bandwidth_temp in bandwidth_list
    ]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf_accelerate(source,
                       target,
                       kernel_mul=2.0,
                       kernel_num=5,
                       fix_sigma=None):
    '''
    Learning Transferable Features with Deep Adaptation Networks
    https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DAN/mmd.py
    '''
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source,
                         target,
                         kernel_mul=2.0,
                         kernel_num=5,
                         fix_sigma=None):
    '''
    Learning Transferable Features with Deep Adaptation Networks
    https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DAN/mmd.py
    '''
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    # actually torch.mean(XY) == torch.mean(YX)
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss[0]


class LMMD_loss(nn.Module):
    '''
    Deep Subdomain Adaptation Network for Image Classification
    https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py
    '''

    def __init__(self,
                 class_num=31,
                 kernel_type='rbf',
                 kernel_mul=2.0,
                 kernel_num=5,
                 fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self,
                        source,
                        target,
                        kernel_mul=2.0,
                        kernel_num=5,
                        fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (
                n_samples**2 - n_samples)
        bandwidth /= kernel_mul**(kernel_num // 2)
        bandwidth_list = [
            bandwidth * (kernel_mul**i) for i in range(kernel_num)
        ]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS) + torch.sum(
            weight_tt * TT) - torch.sum(2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, class_num=31):
        s_batch_size = s_label.size()[0]
        t_batch_size = t_label.size()[0]
        s_sca_label = s_label.cpu().data.max(1)[1].numpy()
        s_vec_label = s_label.cpu().data.numpy()
        # s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        # avoid the error that division by zero
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        t_mask_arr = np.zeros((t_batch_size, class_num))
        t_mask_arr[:, index] = 1
        s_mask_arr = np.zeros((s_batch_size, class_num))
        s_mask_arr[:, index] = 1
        t_vec_label = t_vec_label * t_mask_arr
        s_vec_label = s_vec_label * s_mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])

        # all shape of the returned weights are two dimensional matrix (len(source_label)*len(source_label))
        return weight_ss.astype('float32'), weight_tt.astype(
            'float32'), weight_st.astype('float32')


def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))


def get_train_dataloader(normal_layer,
                         train_batch_size,
                         percentile=False,
                         using_dataset='all'):
    train_dataloader_set = []
    for i in range(4):
        train_dataset = Dataset_Train(
            split_type=i,
            normal_layer=normal_layer,
            percentile=percentile,
            using_dataset=using_dataset)
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True)
        train_dataloader_set.append(train_dataloader)
    if percentile:
        return train_dataloader_set, train_dataset.percentile
    return train_dataloader_set


def get_target_train_dataloader(train_batch_size,
                                dataset_num=None,
                                dataset=None):
    Darts = Dataset_Darts(dataset_num, dataset)
    dataloader_darts = DataLoader(
        Darts, batch_size=train_batch_size, shuffle=True)
    return dataloader_darts


class DirectedGraphConvolution(nn.Module):
    '''
    Wei Wen, Hanxiao Liu, Hai Li, Yiran Chen, Gabriel Bender, Pieter-Jan Kindermans. "Neural Predictor for Neural
    Architecture Search". arXiv:1912.00848.
    https://github.com/ultmaster/neuralpredictor.pytorch
    '''

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)

    def forward(self, inputs, adj):
        norm_adj = normalize_adj(adj)
        output1 = F.relu(
            torch.matmul(norm_adj, torch.matmul(inputs, self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        output2 = F.relu(
            torch.matmul(inv_norm_adj, torch.matmul(inputs, self.weight2)))
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NeuralPredictor(nn.Module):
    '''
    Wei Wen, Hanxiao Liu, Hai Li, Yiran Chen, Gabriel Bender, Pieter-Jan Kindermans. "Neural Predictor for Neural
    Architecture Search". arXiv:1912.00848.
    https://github.com/ultmaster/neuralpredictor.pytorch
    '''

    def __init__(self,
                 initial_hidden=6,
                 gcn_hidden=144,
                 gcn_layers=5,
                 linear_hidden=128):
        super().__init__()
        self.gcn = [
            DirectedGraphConvolution(initial_hidden if i == 0 else gcn_hidden,
                                     gcn_hidden) for i in range(gcn_layers)
        ]
        self.gcn = nn.ModuleList(self.gcn)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(gcn_hidden, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

    def forward(self, inputs):
        numv, adj, out = inputs['num_vertices'], inputs['adjacency'], inputs[
            'operations']
        gs = adj.size(1)  # graph node number
        adj_with_diag = normalize_adj(
            adj +
            torch.eye(gs, device=adj.device))  # assuming diagonal is not 1
        for layer in self.gcn:
            out = layer(out, adj_with_diag)
        out = graph_pooling(out, numv)
        out = self.fc1(out)
        # out = self.dropout(out)
        # out = self.fc2(out).view(-1)
        return out


class DomainAdaptationPredictor(nn.Module):

    def __init__(self, percentile, gcn_hidden):
        super(DomainAdaptationPredictor, self).__init__()
        self.NeuralPredictor = NeuralPredictor(gcn_hidden=gcn_hidden)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(128, 1, bias=False)
        self.percentile = percentile

    def forward(self,
                source,
                target,
                s_label,
                K,
                kernel_type='rbf',
                loss_type='lmmd'):
        loss = 0
        source = self.NeuralPredictor(source)
        if self.training == True:
            target = self.NeuralPredictor(target)
            t_label = self.fc(target).view(-1)
            lmmd_loss = LMMD_loss(class_num=K, kernel_type=kernel_type)
            K_percentile = self.percentile[K - 1]
            s_label = self.one_hot_classification(K_percentile, s_label)
            t_label = self.one_hot_classification(K_percentile, t_label)
            s_label = torch.from_numpy(s_label)
            t_label = torch.from_numpy(t_label)

            if loss_type == 'lmmd':
                loss += lmmd_loss.get_loss(source, target, s_label, t_label)
            # elif loss_type == 'coral':
            #     loss += coral.CORAL(source, target)
            #     loss = [loss]
            else:
                raise ValueError('loss_type error!')
            # loss += mmd.mmd_rbf_noaccelerate(source, target)
            # if loss < 0:
            #     print()
        source = self.dropout(source)
        source = self.fc(source).view(-1)

        return source, loss

    def one_hot_classification(self, K_percentile, labels):

        def classification(label, K_percentile):
            for j, percentile in enumerate(K_percentile):
                if j == len(K_percentile) - 1:
                    return j
                if (label < K_percentile[j + 1]) and (percentile < label):
                    return j

        batch_size = labels.size()[0]
        one_hot_label = np.zeros((batch_size, len(K_percentile)), dtype=int)
        for i, label in enumerate(labels):
            class_num = classification(label, K_percentile)
            one_hot_label[i][class_num] = 1
        return one_hot_label


class AdvDomainAdaptationPredictor(nn.Module):

    def __init__(self, percentile, gcn_hidden):
        super(AdvDomainAdaptationPredictor, self).__init__()
        self.NeuralPredictor = NeuralPredictor(gcn_hidden=gcn_hidden)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(128, 1, bias=False)
        # no uses of percentile
        self.percentile = percentile
        self.domain_classifier = DANN(input_size=128)

    def forward(self, source, target):
        loss = 0
        source = self.NeuralPredictor(source)
        source_batch_size = source.shape[0]
        if self.training == True:
            target = self.NeuralPredictor(target)
            loss_domain = torch.nn.NLLLoss()
            loss_domain = loss_domain.cuda()

            # add source domain loss
            # domain label = 0
            domain_label = torch.zeros(source_batch_size)
            domain_label = domain_label.long()
            domain_label = domain_label.cuda()

            domain_output = self.domain_classifier(source)
            loss += loss_domain(domain_output, domain_label)

            # add target domain loss
            # domain label = 1
            target_batch_size = target.shape[0]
            domain_label = torch.ones(target_batch_size)
            domain_label = domain_label.long()
            domain_label = domain_label.cuda()

            domain_output = self.domain_classifier(target)
            loss += loss_domain(domain_output, domain_label)

        source = self.dropout(source)
        source = self.fc(source).view(-1)
        return source, loss


class GCN_predictor():

    def __init__(self,
                 percentile,
                 gcn_hidden=144,
                 speed='cos',
                 K=3,
                 is_adv=False):
        if is_adv == False:
            self.normal_predictor0 = DomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden)
            self.normal_predictor1 = DomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden)
            self.reduction_predictor0 = DomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden)
            self.reduction_predictor1 = DomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden)
        else:
            # using Adversarial loss
            self.normal_predictor0 = AdvDomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden)
            self.normal_predictor1 = AdvDomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden)
            self.reduction_predictor0 = AdvDomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden)
            self.reduction_predictor1 = AdvDomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden)
        self.speed = speed
        self._K = K
        self._is_adv = is_adv
