import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce


class BayesianLinear(nn.Module):
    """Bayesian Linear Layer."""

    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        # Parameters for mean
        self.mean = nn.Parameter(torch.Tensor(out_features, in_features))
        # Parameters for variance (rho)
        self.rho = nn.Parameter(torch.Tensor(out_features, in_features))
        # Standard deviation (sigma) will be derived from rho
        self.register_buffer('eps', torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mean, a=math.sqrt(5))
        nn.init.constant_(self.rho, -5)  # It makes initial sigma close to 0

    def forward(self, input):
        # Reparameterization trick
        self.eps.data.normal_()
        sigma = torch.log1p(torch.exp(self.rho))
        # W = mean + sigma * epsilon
        W = self.mean + sigma * self.eps

        return F.linear(input, W)


class BayesianMLP(nn.Module):
    """Multilayer Perceptron with Bayesian Linear Layer."""

    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bayesian_fc = BayesianLinear(hidden_size,
                                          hidden_size)  # Bayesian layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.bayesian_fc(x))  # Pass through the Bayesian layer
        x = self.fc2(x)
        return x


class PreNormResidual(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim), nn.GELU(), nn.Dropout(dropout),
        dense(inner_dim, dim), nn.Dropout(dropout))


class BaysianMLPMixer(nn.Module):

    def __init__(
            self,
            input_dim=83 * 3,  # layerwise zc
            sequence_length=256,
            patch_size=16,
            dim=512,
            depth=4,
            emb_out_dim=14,
            expansion_factor=4,
            expansion_factor_token=0.5,
            dropout=0.1):
        super(BaysianMLPMixer, self).__init__()
        assert sequence_length % patch_size == 0, 'sequence length must be divisible by patch size'
        num_patches = sequence_length // patch_size

        self.project = nn.Linear(input_dim, sequence_length)

        self.patch_rearrange = Rearrange('b (l p) -> b l (p)', p=patch_size)
        self.patch_to_embedding = BayesianLinear(patch_size, dim)
        self.mixer_blocks = nn.ModuleList([
            nn.Sequential(
                PreNormResidual(dim, FeedForward(dim, expansion_factor,
                                                 dropout)),
                PreNormResidual(
                    dim, FeedForward(dim, expansion_factor_token, dropout)))
            for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(dim)
        self.reduce = Reduce('b l c -> b c', 'mean')
        self.head = BayesianLinear(dim, dim)

        # regressor
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(dim, dim // 2, bias=False)
        self.fc2 = nn.Linear(dim // 2, 1, bias=False)

    def forward(self, x):
        x = self.project(x)
        x = self.patch_rearrange(x)
        x = self.patch_to_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.reduce(x)
        x = self.head(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
