# Run2: only zcp

import collections.abc
from functools import partial
from itertools import repeat

import torch.nn as nn

from .pinat_run1 import Encoder, graph_pooling


class PINATModel2(nn.Module):
    """Only zcp is needed."""

    def __init__(self,
                 adj_type,
                 n_src_vocab,
                 d_word_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 pad_idx=None,
                 pos_enc_dim=7,
                 linear_hidden=80,
                 pine_hidden=256,
                 bench='101',
                 dropout=0.1,
                 zcp_embedder_dims=[128, 128],
                 gating=False):
        super(PINATModel2, self).__init__()

        # backone
        self.bench = bench
        self.adj_type = adj_type
        # regressor
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(d_word_vec, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

        # zcp embedder
        self.zcp_embedder_dims = zcp_embedder_dims
        self.zcp_embedder = []
        mid_zcp_dim = 38
        # 13 zcs
        for zcp_emb_dim in self.zcp_embedder_dims:  # [128, 128]
            self.zcp_embedder.append(
                nn.Sequential(
                    nn.Linear(mid_zcp_dim, zcp_emb_dim),
                    nn.ReLU(inplace=False), nn.Dropout(p=dropout)))
            mid_zcp_dim = zcp_emb_dim
        self.zcp_embedder.append(nn.Linear(mid_zcp_dim, d_word_vec))
        self.zcp_embedder = nn.Sequential(*self.zcp_embedder)

        if gating is True:
            self.gating = nn.Sequential(
                nn.Linear(d_word_vec, d_word_vec), nn.ReLU(inplace=False),
                nn.Dropout(p=dropout), nn.Linear(d_word_vec, d_word_vec),
                nn.Sigmoid())

    def forward(self, inputs):
        # zc embedder
        zc_embed = self.zcp_embedder(inputs['zcp'])
        out = zc_embed

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(
            nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerGateBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=(0.5, 4.0),
                 mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()

        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]

        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(
            dim, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(
            dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.mlp_tokens(self.norm1(x))
        x = x + self.mlp_channels(self.norm2(x))
        return x


class PINATModel3(nn.Module):
    """
        Only zcp is needed.
    """

    def __init__(self,
                 adj_type,
                 n_src_vocab,
                 d_word_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 pad_idx=None,
                 pos_enc_dim=7,
                 linear_hidden=80,
                 pine_hidden=256,
                 bench='101',
                 dropout=0.1,
                 zcp_embedder_dims=[256, 512, 1024, 2048],
                 gating=False):
        super(PINATModel3, self).__init__()

        # backone
        self.bench = bench
        self.adj_type = adj_type
        # regressor
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(d_word_vec, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

        # zcp embedder
        self.zcp_embedder_dims = zcp_embedder_dims
        self.zcp_embedder = []
        mid_zcp_dim = 98 * 3

        for zcp_emb_dim in self.zcp_embedder_dims:  # [128, 128]
            self.zcp_embedder.append(
                nn.Sequential(
                    nn.Linear(mid_zcp_dim, zcp_emb_dim),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=dropout),
                ))
            mid_zcp_dim = zcp_emb_dim
        self.zcp_embedder.append(nn.Linear(mid_zcp_dim, d_word_vec))
        self.zcp_embedder = nn.Sequential(*self.zcp_embedder)

        self.gate_block = MixerGateBlock(
            d_word_vec,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.1)

    def forward(self, inputs):
        # zc embedder
        zc_embed = self.zcp_embedder(inputs['zcp_layerwise'])

        # attention machenism
        out = self.gate_block(zc_embed)
        out = zc_embed

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out


class PINATModel4(nn.Module):
    """
        PINATModel + zcp embedding (naive embedder)
    """

    def __init__(self,
                 adj_type,
                 n_src_vocab,
                 d_word_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 pad_idx=None,
                 pos_enc_dim=7,
                 linear_hidden=80,
                 pine_hidden=256,
                 bench='101',
                 dropout=0.1,
                 zcp_embedder_dims=[128, 128]):
        super(PINATModel4, self).__init__()

        # backone
        self.bench = bench
        self.adj_type = adj_type
        self.encoder = Encoder(
            n_src_vocab,
            d_word_vec,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_model,
            d_inner,
            pad_idx,
            pos_enc_dim=pos_enc_dim,
            dropout=0.1,
            pine_hidden=pine_hidden,
            bench=bench)

        # regressor
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(d_word_vec, linear_hidden, bias=False)
        self.fc2 = nn.Linear(linear_hidden, 1, bias=False)

        # zcp embedder
        self.zcp_embedder_dims = zcp_embedder_dims
        self.zcp_embedder = []
        mid_zcp_dim = 98 * 3 # for nb201

        for zcp_emb_dim in self.zcp_embedder_dims:  # [128, 128]
            self.zcp_embedder.append(
                nn.Sequential(
                    nn.Linear(mid_zcp_dim, zcp_emb_dim),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=dropout),
                ))
            mid_zcp_dim = zcp_emb_dim
        self.zcp_embedder.append(nn.Linear(mid_zcp_dim, d_word_vec))
        self.zcp_embedder = nn.Sequential(*self.zcp_embedder)

        self.gate_block = MixerGateBlock(
            d_word_vec,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.1)

    def forward(self, inputs):
        # get arch topology
        numv = inputs['num_vertices']
        assert self.adj_type == 'adj_lapla'
        adj_matrix = inputs['lapla'].float()  # bs, 7, 7
        edge_index_list = []
        for edge_num, edge_index in zip(
                inputs['edge_num'],  # bs
                inputs['edge_index_list']):  # bs, 2, 9
            edge_index_list.append(edge_index[:, :edge_num])

        # backone feature
        out = self.encoder(
            src_seq=inputs['features'],  # bs, 7
            pos_seq=adj_matrix.float(),  # bs, 7, 7
            operations=inputs['operations'].squeeze(0),  # bs, 7, 5
            num_nodes=numv,
            edge_index_list=edge_index_list)

        # regressor forward
        out = graph_pooling(out, numv)

        # zc embedder
        zc_embed = self.zcp_embedder(inputs['zcp_layerwise'])
        zc_embed = self.gate_block(zc_embed)
        out += zc_embed

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out
