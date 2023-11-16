# Author: Yang Liu @ Abacus.ai
# This is an implementation of gcn predictor for NAS from the paper:
# Wen et al., 2019. Neural Predictor for Neural Architecture Search

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))


def accuracy_mse(prediction, target, scale=100.0):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


class DirectedGraphConvolution(nn.Module):
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
        output1 = F.relu(torch.matmul(
            norm_adj, torch.matmul(inputs, self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        output2 = F.relu(torch.matmul(
            inv_norm_adj, torch.matmul(inputs, self.weight2)))
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


class NeuralPredictorModel(nn.Module):
    def __init__(
        self, initial_hidden=-1, gcn_hidden=144, gcn_layers=4, linear_hidden=128
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
        self.proj = nn.Linear(5, 4)

    def forward(self, inputs):
        numv, adj, out = (
            inputs['num_vertices'],
            inputs['adjacency'],
            inputs['operations'],
        )  # 7 6 5
        gs = adj.size(1)  # graph node number

        adj_with_diag = normalize_adj(
            adj + torch.eye(gs, device=adj.device)
        )  # assuming diagonal is not 1
        out = self.proj(out)  # 7 6 6
        for layer in self.gcn:
            out = layer(out, adj_with_diag)  # 7 4 4
        out = graph_pooling(out, numv)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out


if __name__ == '__main__':
    gcn_hidden = 144
    batch_size = 7

    ss_type = 'nasbench201'
    initial_hidden = 4

    predictor = NeuralPredictorModel(
        initial_hidden=initial_hidden, gcn_hidden=gcn_hidden
    )
    if torch.cuda.is_available():
        predictor = predictor.cuda()

    from piconas.datasets.predictor.nb201_dataset import Nb201DatasetPINAT

    test_set = Nb201DatasetPINAT(
        split='all', data_type='test', data_set='cifar10')

    loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    for batch in loader:
        input = batch
        score = predictor(input)
