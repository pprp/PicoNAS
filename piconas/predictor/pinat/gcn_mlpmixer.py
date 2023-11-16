import torch
import torch.nn as nn
import torch.nn.functional as F

from piconas.datasets.predictor.nb201_dataset import Nb201DatasetPINAT
from piconas.predictor.pinat.mlpmixer import FeedForward, PreNormResidual


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


class NeuralPredictorMLPMixer(nn.Module):
    def __init__(
        self,
        initial_hidden=4,
        gcn_hidden=144,
        gcn_layers=4,
        linear_hidden=128,
        input_dim=294,
        dim=512,
        depth=4,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=0.1,
    ):
        super(NeuralPredictorMLPMixer, self).__init__()
        self.initial_hidden = initial_hidden
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
        self.proj = nn.Linear(5, initial_hidden)  # Adjusted to initial_hidden

        # MLP-Mixer
        self.project = nn.Linear(input_dim, dim)

        self.mixer_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    PreNormResidual(dim, FeedForward(
                        dim, expansion_factor, dropout)),
                    PreNormResidual(
                        dim, FeedForward(dim, expansion_factor_token, dropout)
                    ),
                )
                for _ in range(depth)
            ]
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, gcn_hidden)

    def forward(self, inputs):
        numv, adj, out = (
            inputs['num_vertices'],
            inputs['adjacency'],
            inputs['operations'],
        )

        # Assuming the first 4 nodes are the relevant ones
        out = out[:, : self.initial_hidden, :]

        gs = adj.size(1)  # graph node number
        adj_with_diag = normalize_adj(adj + torch.eye(gs, device=adj.device))
        out = self.proj(out)
        for layer in self.gcn:
            out = layer(out, adj_with_diag)
        out = graph_pooling(out, numv)

        # MLP-Mixer
        x = inputs['zcp_layerwise']
        x = self.project(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.fc(x)

        out = self.fc1(out + x)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out


if __name__ == '__main__':
    model = NeuralPredictorMLPMixer()

    if torch.cuda.is_available():
        model = model.cuda()

    # Assuming Nb201DatasetPINAT is defined elsewhere
    test_set = Nb201DatasetPINAT(
        split='all', data_type='test', data_set='cifar10')

    loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=0, drop_last=False
    )

    for batch in loader:
        input = batch
        score = model(input)
        print(score)
