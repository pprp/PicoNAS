import argparse
import math
import os
import time

import coral
import mmd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DANN import DANN
from dataset_matrix import Dataset_Darts, Dataset_Train
from torch.utils.data import DataLoader
from utils import AverageMeterGroup, convert_to_genotype, get_logger, to_cuda


def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))


def get_train_dataloader(
    normal_layer, train_batch_size, percentile=False, using_dataset='all'
):
    train_dataloader_set = []
    for i in range(4):
        train_dataset = Dataset_Train(
            split_type=i,
            normal_layer=normal_layer,
            percentile=percentile,
            using_dataset=using_dataset,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )
        train_dataloader_set.append(train_dataloader)
    if percentile:
        return train_dataloader_set, train_dataset.percentile
    return train_dataloader_set


def get_target_train_dataloader(train_batch_size, dataset_num=None, dataset=None):
    Darts = Dataset_Darts(dataset_num, dataset)
    dataloader_darts = DataLoader(Darts, batch_size=train_batch_size, shuffle=True)
    return dataloader_darts


class DirectedGraphConvolution(nn.Module):
    """
    Wei Wen, Hanxiao Liu, Hai Li, Yiran Chen, Gabriel Bender, Pieter-Jan Kindermans. "Neural Predictor for Neural
    Architecture Search". arXiv:1912.00848.
    https://github.com/ultmaster/neuralpredictor.pytorch
    """

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
        output1 = F.relu(torch.matmul(norm_adj, torch.matmul(inputs, self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        output2 = F.relu(torch.matmul(inv_norm_adj, torch.matmul(inputs, self.weight2)))
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

    def __repr__(self):
        return (
            self.__class__.__name__
            + ' ('
            + str(self.in_features)
            + ' -> '
            + str(self.out_features)
            + ')'
        )


class NeuralPredictor(nn.Module):
    """
    Wei Wen, Hanxiao Liu, Hai Li, Yiran Chen, Gabriel Bender, Pieter-Jan Kindermans. "Neural Predictor for Neural
    Architecture Search". arXiv:1912.00848.
    https://github.com/ultmaster/neuralpredictor.pytorch
    """

    def __init__(
        self, initial_hidden=6, gcn_hidden=144, gcn_layers=5, linear_hidden=128
    ):
        super().__init__()
        self.gcn = [
            DirectedGraphConvolution(
                initial_hidden if i == 0 else gcn_hidden, gcn_hidden
            )
            for i in range(gcn_layers)
        ]
        self.gcn = nn.ModuleList(self.gcn)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(gcn_hidden, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

    def forward(self, inputs):
        numv, adj, out = (
            inputs['num_vertices'],
            inputs['adjacency'],
            inputs['operations'],
        )
        gs = adj.size(1)  # graph node number
        adj_with_diag = normalize_adj(
            adj + torch.eye(gs, device=adj.device)
        )  # assuming diagonal is not 1
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

    def forward(self, source, target, s_label, K, kernel_type='rbf', loss_type='lmmd'):
        loss = 0
        source = self.NeuralPredictor(source)
        if self.training == True:
            target = self.NeuralPredictor(target)
            t_label = self.fc(target).view(-1)
            lmmd_loss = mmd.LMMD_loss(class_num=K, kernel_type=kernel_type)
            K_percentile = self.percentile[K - 1]
            s_label = self.one_hot_classification(K_percentile, s_label)
            t_label = self.one_hot_classification(K_percentile, t_label)
            s_label = torch.from_numpy(s_label)
            t_label = torch.from_numpy(t_label)

            if loss_type == 'lmmd':
                loss += lmmd_loss.get_loss(source, target, s_label, t_label)
            elif loss_type == 'coral':
                loss += coral.CORAL(source, target)
                loss = [loss]
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


class GCN_predictor:
    def __init__(self, percentile, gcn_hidden=144, speed='cos', K=3, is_adv=False):
        if is_adv == False:
            self.normal_predictor0 = DomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden
            )
            self.normal_predictor1 = DomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden
            )
            self.reduction_predictor0 = DomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden
            )
            self.reduction_predictor1 = DomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden
            )
        else:
            # using Adversarial loss
            self.normal_predictor0 = AdvDomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden
            )
            self.normal_predictor1 = AdvDomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden
            )
            self.reduction_predictor0 = AdvDomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden
            )
            self.reduction_predictor1 = AdvDomainAdaptationPredictor(
                percentile, gcn_hidden=gcn_hidden
            )
        self.speed = speed
        self._K = K
        self._is_adv = is_adv

    def split_target_dataset(self, batch):
        batch_set = []
        adjacency = batch['adjacency']
        num_vertices = batch['num_vertices']
        operations = batch['operations']
        for i in range(4):
            batch_set.append(
                {
                    'adjacency': adjacency[i],
                    'num_vertices': num_vertices[i],
                    'operations': operations[i],
                }
            )

        return batch_set

    def train(
        self,
        data_loader_set,
        target_data_loader,
        assistant_data_loader,
        epochs=50,
        init_lr=2e-3,
        wd=1e-3,
        train_print_freq=10,
        assistant_rate=0.15,
        loss_type='lmmd',
        kernel_type='rbf',
    ):
        logger = get_logger()
        # assign K
        K = self._K
        # calculate assistant epochs
        assistant_epochs = epochs * assistant_rate
        # if do not use assistant_data_loader
        if assistant_data_loader is None:
            logger.info('Do not use assistant dataloader!!!')
            assistant_data_loader = target_data_loader

        net_set = [
            self.normal_predictor0,
            self.normal_predictor1,
            self.reduction_predictor0,
            self.reduction_predictor1,
        ]
        assert len(net_set) == len(data_loader_set)
        i = -1
        for net, data_loader in zip(net_set, data_loader_set):
            i += 1
            criterion = nn.MSELoss()
            net.cuda()
            optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=wd)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            net.train()
            for epoch in range(epochs):
                # calculate k first
                # k = 2
                if self.speed == 'cos':
                    k = K - math.floor(math.cos((epoch + 1) / epochs * math.pi / 2) * K)
                elif self.speed == 'sin':
                    k = (
                        math.floor(
                            math.cos((epochs - (epoch + 1)) / epochs * math.pi / 2) * K
                        )
                        + 1
                    )
                else:
                    # linear
                    k = math.floor((epoch + 1) / epochs * K) + 1
                logger.info('Epoch: {}, k: {}'.format(epoch + 1, k))

                meters = AverageMeterGroup()
                lr = optimizer.param_groups[0]['lr']
                # determine whether to use assistant space
                if epoch < assistant_epochs:
                    logger.info('Using assistant data loader')
                    target_iter = iter(assistant_data_loader)
                else:
                    logger.info('Using target data loader')
                    target_iter = iter(target_data_loader)

                for step, batch in enumerate(data_loader):
                    batch = to_cuda(batch)
                    s_label = batch['val_acc'].to(torch.float)
                    try:
                        target_data = target_iter.next()
                    except Exception:
                        target_iter = iter(target_data_loader)
                        target_data = target_iter.next()
                    target_data_set = self.split_target_dataset(target_data)
                    target_data_set = to_cuda(target_data_set)

                    if self._is_adv == False:
                        # not use Adversarial loss
                        predict, mmd_loss = net(
                            batch,
                            target_data_set[i],
                            s_label,
                            k,
                            loss_type=loss_type,
                            kernel_type=kernel_type,
                        )
                        mmd_loss = mmd_loss[0]
                    else:
                        # using Adversarial loss
                        predict, domain_loss = net(batch, target_data_set[i])
                        # maybe rename the mmd_loss
                        mmd_loss = domain_loss
                    optimizer.zero_grad()
                    loss = criterion(predict, s_label)
                    lambd = 2 / (1 + math.exp(-10 * epoch / epochs)) - 1
                    # if mmd_loss < 1e-6:
                    #     # if mmd loss is too small, ignore it
                    #     lambd = 0
                    # lambd = 0
                    loss += lambd * mmd_loss
                    # if torch.isnan(loss):
                    #     break
                    loss.backward()
                    optimizer.step()

                    meters.update(
                        {'loss': loss.item(), 'DA_loss': mmd_loss}, n=s_label.size(0)
                    )
                    if (
                        train_print_freq and step % train_print_freq == 0
                    ) or step + 1 == len(data_loader):
                        logger.info(
                            'Epoch [%d/%d] Step [%d/%d] lr = %.3e  %s',
                            epoch + 1,
                            epochs,
                            step + 1,
                            len(data_loader),
                            lr,
                            meters,
                        )
                lr_scheduler.step()

    def train_without_domain_adaptation(
        self, data_loader_set, label, epochs=50, init_lr=5e-1, wd=1e-3
    ):
        net_set = [
            self.normal_predictor0,
            self.normal_predictor1,
            self.reduction_predictor0,
            self.reduction_predictor1,
        ]
        label = torch.tensor(label)
        label = label.cuda()
        i = -1
        print('CUDA available:', torch.cuda.is_available())
        tmp_iter = iter(data_loader_set)
        batch = tmp_iter.next()
        for net in net_set:
            i += 1
            criterion = nn.MSELoss()
            net.cuda()
            optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=wd)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            net.train()
            for epoch in range(epochs):
                meters = AverageMeterGroup()
                lr = optimizer.param_groups[0]['lr']

                batch_set = self.split_target_dataset(batch)
                for j in range(4):
                    batch_set[j] = to_cuda(batch_set[j])

                target, s_label, K = None, None, None
                predict, mmd_loss = net(batch_set[i], target, s_label, K)
                optimizer.zero_grad()
                loss = criterion(predict, label)
                loss.backward()
                optimizer.step()
                # seconde train
                # if i==0:
                #     se_i = 1
                # elif i==1:
                #     se_i = 0
                # elif i==2:
                #     se_i = 3
                # else:
                #     se_i = 2
                # predict, mmd_loss = net(batch_set[se_i], target, s_label, K)
                # optimizer.zero_grad()
                # loss = criterion(predict, label)
                # loss.backward()
                # optimizer.step()
                print(epoch, loss)

            lr_scheduler.step()

    def predict(self, pred_data_loader, normal_layer):
        normal_rate = normal_layer / (normal_layer + 2)
        net_set = [
            self.normal_predictor0,
            self.normal_predictor1,
            self.reduction_predictor0,
            self.reduction_predictor1,
        ]
        predict_ = []

        with torch.no_grad():
            for step, batch in enumerate(pred_data_loader):
                # recombine batch
                batch_set = self.split_target_dataset(batch)
                for i in range(4):
                    batch_set[i] = to_cuda(batch_set[i])

                predict_list = []
                for j, net in enumerate(net_set):
                    net.cuda()
                    net.eval()
                    # no target in evaluation stage
                    target, s_label, K = None, None, None
                    if self._is_adv == False:
                        predict, mmd_loss = net(batch_set[j], target, s_label, K)
                    else:
                        # using adversarial network
                        predict, _ = net(batch_set[j], target)
                    predict = predict.cpu().detach().numpy()
                    predict_list.append(predict)

                # weighted sum
                weighted_sum_y = (
                    normal_rate
                    * (np.array(predict_list[0]) + np.array(predict_list[1]))
                    / 2
                    + (1 - normal_rate)
                    * (np.array(predict_list[2]) + np.array(predict_list[3]))
                    / 2
                )
                predict_.extend(weighted_sum_y)

        assert len(predict_) == len(pred_data_loader.dataset)
        return predict_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN_predictor')
    parser.add_argument(
        '--cifarORimage',
        type=str,
        default='cifar',
        choices=['cifar', 'image'],
        help='search for cells on cifar10 or on imagenet',
    )
    parser.add_argument('--train_batch_size', default=1000, type=int)
    parser.add_argument('--test_batch_size', default=100000, type=int)
    parser.add_argument('--gpu_id', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # train
    localtime = time.asctime(time.localtime(time.time()))
    print('start loading training data:{}'.format(localtime))
    if args.cifarORimage == 'cifar':
        normal_layer = 18
    elif args.cifarORimage == 'image':
        normal_layer = 12
    else:
        raise ValueError("the normal_type should be chosen from ['cifar', 'image']")
    train_dataloader_set, percentile = get_train_dataloader(
        normal_layer, args.train_batch_size, percentile=True
    )
    # darts_save_path = 'path/darts_dataset.pkl'
    # if os.path.exists(darts_save_path):
    #     print('load dataset.')
    #     with open(darts_save_path, 'rb') as file:
    #         saved_darts_dataset = pickle.load(file)
    #     Darts = Dataset_Darts(dataset_num=1e6, dataset=saved_darts_dataset.dataset)
    # else:
    Darts = Dataset_Darts()
    target_dataloader = DataLoader(
        Darts, batch_size=args.train_batch_size, shuffle=True
    )
    # assistant dataloader
    ### Maybe need to add a function saving tiny darts
    Tiny_darts = Dataset_Darts(dataset_num=5e4, dataset_type='tiny')
    assistant_dataloader = DataLoader(
        Tiny_darts, batch_size=args.train_batch_size, shuffle=True
    )
    predictor = GCN_predictor(percentile)

    localtime = time.asctime(time.localtime(time.time()))
    print('end loading training data, start training:{}'.format(localtime))
    # If you do not want to use the assistant dataloader
    predictor.train(train_dataloader_set, target_dataloader, assistant_dataloader)

    # prediction
    localtime = time.asctime(time.localtime(time.time()))
    print('start loading darts data:{}'.format(localtime))
    # Darts = Dataset_Darts()
    dataloader_darts = DataLoader(Darts, batch_size=args.test_batch_size, shuffle=False)

    localtime = time.asctime(time.localtime(time.time()))
    print('start predicting:{}'.format(localtime))
    pred_y = predictor.predict(dataloader_darts, normal_layer)
    localtime = time.asctime(time.localtime(time.time()))
    print('end predicting:{}'.format(localtime))

    K = 3
    best_index_list = np.argsort(pred_y)[-K:]
    print('Top {} architectures:'.format(K))
    for best_index in best_index_list:
        integer_genotype = Darts.dataset.dataset[best_index]
        genotype = convert_to_genotype(integer_genotype)
        print(genotype)
