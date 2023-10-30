# Run1: pinat + zcp
import collections.abc
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data

from piconas.predictor.pinat.gatset_conv import GATSetConv_v5 as GATConv


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    num_vertices = num_vertices.unsqueeze(-1).expand_as(out)

    return torch.div(out, num_vertices)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self,
                 sa_heads,
                 d_model,
                 d_k,
                 d_v,
                 pine_hidden=256,
                 dropout=0.1,
                 pine_heads=2,
                 bench='101'):
        super().__init__()

        sa_heads = 2
        self.n_head = sa_heads
        self.d_k = d_k
        self.d_v = d_v

        # standard multihead attention
        self.w_qs = nn.Linear(d_model, sa_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, sa_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, sa_heads * d_v, bias=False)
        self.fc = nn.Linear((sa_heads + pine_heads) * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        # pine structure
        self.conv1 = GATConv(d_model, pine_hidden, heads=pine_heads)
        self.lin1 = torch.nn.Linear(d_model, pine_heads * pine_hidden)
        self.conv2 = GATConv(
            pine_heads * pine_hidden, pine_hidden, heads=pine_heads)
        self.lin2 = torch.nn.Linear(pine_heads * pine_hidden,
                                    pine_heads * pine_hidden)
        self.conv3 = GATConv(
            pine_heads * pine_hidden,
            pine_heads * d_k,
            heads=pine_heads,
            concat=False)
        self.lin3 = torch.nn.Linear(pine_heads * pine_hidden, pine_heads * d_k)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.bench = bench
        if self.bench == '201':
            self.proj_func = nn.Linear(4, 6)

    def to_pyg_batch(self, xs, edge_index_list, num_nodes):
        assert xs.shape[0] == len(edge_index_list)
        assert xs.shape[0] == len(num_nodes)
        data_list = []
        for x, e, n in zip(xs, edge_index_list, num_nodes):
            data_list.append(torch_geometric.data.Data(x=x[:n], edge_index=e))
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        return batch

    def forward(self, q, k, v, edge_index_list, num_nodes, mask=None):
        # PISA
        x = q
        bs = x.shape[0]
        pyg_batch = self.to_pyg_batch(x, edge_index_list, num_nodes)
        x = F.elu(
            self.conv1(pyg_batch.x, pyg_batch.edge_index) +
            self.lin1(pyg_batch.x))
        x = F.elu(self.conv2(x, pyg_batch.edge_index) + self.lin2(x))
        x = self.conv3(x, pyg_batch.edge_index) + self.lin3(x)
        x = x.view(bs, -1, x.shape[-1])
        if self.bench == '201':
            x = x.transpose(1, 2)
            x = self.proj_func(x)
            x = x.transpose(1, 2)

        # SA
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        # self-attention + PISA
        q = torch.cat((x, q), dim=-1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 pine_hidden,
                 dropout=0.1,
                 bench='101'):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            pine_hidden,
            dropout=dropout,
            bench=bench)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self,
                enc_input,
                edge_index_list,
                num_nodes,
                slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input,
            enc_input,
            enc_input,
            edge_index_list,
            num_nodes,
            mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """ An encoder model with self attention mechanism. """

    def __init__(self,
                 n_src_vocab,
                 d_word_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 pad_idx,
                 pos_enc_dim=7,
                 dropout=0.1,
                 n_position=200,
                 bench='101',
                 in_features=5,
                 pine_hidden=256,
                 heads=6,
                 linear_input=80):
        super().__init__()

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.bench = bench
        if self.bench == '101':
            self.embedding_lap_pos_enc = nn.Linear(
                pos_enc_dim, d_word_vec)  # position embedding
        elif self.bench == '201':
            self.pos_map = nn.Linear(pos_enc_dim, n_src_vocab + 1)
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, d_word_vec)
            self.proj_func = nn.Linear(4, 6)
        else:
            raise ValueError('No defined NAS bench.')

        # pine structure
        self.conv1 = GATConv(in_features, pine_hidden, heads=heads)
        self.lin1 = torch.nn.Linear(in_features, heads * pine_hidden)

        self.conv2 = GATConv(heads * pine_hidden, pine_hidden, heads=heads)
        self.lin2 = torch.nn.Linear(heads * pine_hidden, heads * pine_hidden)

        self.conv3 = GATConv(
            heads * pine_hidden, linear_input, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(heads * pine_hidden, linear_input)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(
                d_model,
                d_inner,
                n_head,
                d_k,
                d_v,
                dropout=dropout,
                pine_hidden=pine_hidden,
                bench=bench) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def to_pyg_batch(self, xs, edge_index_list, num_nodes):
        # import pdb; pdb.set_trace()
        assert xs.shape[0] == len(
            edge_index_list), f'{xs.shape[0]}, {len(edge_index_list)}'
        assert xs.shape[0] == len(
            num_nodes), f'{xs.shape[0]}, {len(num_nodes)}'
        data_list = []
        for x, e, n in zip(xs, edge_index_list, num_nodes):
            data_list.append(torch_geometric.data.Data(x=x[:n], edge_index=e))
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        return batch

    def forward(
            self,
            src_seq,  # features: bs, 7
            pos_seq,  # lapla: bs, 7, 7
            operations,  # operations: bs, 7, 5 -> bs, 35
            edge_index_list,  # list with different length tensor
            num_nodes,  # num of node: bs
            src_mask=None):
        # op emb and pos emb
        enc_output = self.src_word_emb(src_seq)
        if self.bench == '101':
            pos_output = self.embedding_lap_pos_enc(
                pos_seq)  # positional embedding
            enc_output += pos_output  # bs, 7, 80
            # enc_output = pos_output
            enc_output = self.dropout(enc_output)
        elif self.bench == '201':
            pos_output = self.pos_map(pos_seq).transpose(1, 2)
            pos_output = self.embedding_lap_pos_enc(pos_output)
            enc_output += pos_output
            enc_output = self.dropout(enc_output)
        else:
            raise ValueError('No defined NAS bench.')

        # PITE
        x = operations  # bs, 7, 5
        bs = operations.shape[0]  # bs=10 for test
        # import pdb; pdb.set_trace()
        pyg_batch = self.to_pyg_batch(x, edge_index_list, num_nodes)
        # pyg_batch.x.shape=[70,5]
        # pyg_batch.edge_index.shape=[2, 84]
        x = F.elu(
            self.conv1(pyg_batch.x, pyg_batch.edge_index) +
            self.lin1(pyg_batch.x))
        x = F.elu(self.conv2(x, pyg_batch.edge_index) + self.lin2(x))
        x = self.conv3(x, pyg_batch.edge_index) + self.lin3(x)
        x = x.view(bs, -1, x.shape[-1])
        if self.bench == '201':
            x = x.transpose(1, 2)
            x = self.proj_func(x)
            x = x.transpose(1, 2)
        enc_output += self.dropout(x)
        enc_output = self.layer_norm(enc_output)

        # backone forward for n_layers (3)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, edge_index_list, num_nodes, slf_attn_mask=src_mask)

        return enc_output


class PINATModel1(nn.Module):
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
        super(PINATModel1, self).__init__()

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
        mid_zcp_dim = 13
        for zcp_emb_dim in self.zcp_embedder_dims:
            self.zcp_embedder.append(
                nn.Sequential(
                    nn.Linear(mid_zcp_dim, zcp_emb_dim),
                    nn.ReLU(inplace=False), nn.Dropout(p=dropout)))
            mid_zcp_dim = zcp_emb_dim
        self.zcp_embedder.append(nn.Linear(mid_zcp_dim, d_word_vec))
        self.zcp_embedder = nn.Sequential(*self.zcp_embedder)

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
        zc_embed = self.zcp_embedder(inputs['zcp'])
        out += zc_embed

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out


from .pinat_model import Encoder, graph_pooling


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

        if bench == '101':
            mid_zcp_dim = 83 * 3 # for nb101
        elif bench == '201':
            mid_zcp_dim = 98 * 3  # for nb201

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
        if bench == '101':
            mid_zcp_dim = 83 * 3 # for nb101 
        elif bench == '201':
            mid_zcp_dim = 98 * 3  # for nb201

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
        assert self.adj_type == 'adj_lapla', f'only support adj_lapla but got {self.adj_type}'
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


class PINATModel5(nn.Module):
    """
        PINATModel5 + zcp embedding (naive embedder)
        PINATModel4 is a small size model with [128,128]
        Here We use large Model
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
                 zcp_embedder_dims=[256, 512, 1024, 2048]):
        super(PINATModel5, self).__init__()

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
        if bench == '101':
            mid_zcp_dim = 83 * 3 # for nb101
        elif bench == '201':
            mid_zcp_dim = 98 * 3  # for nb201

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
        assert self.adj_type == 'adj_lapla', f'only support adj_lapla but got {self.adj_type}'
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



class Encoder6(nn.Module):
    """ An encoder model with self attention mechanism. """

    def __init__(self,
                 n_src_vocab,
                 d_word_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 pad_idx,
                 pos_enc_dim=7,
                 dropout=0.1,
                 n_position=200,
                 bench='101',
                 in_features=5,
                 pine_hidden=256,
                 heads=6,
                 linear_input=512, #80, 
                 zcp_embedder_dims=[256, 512, 1024, 2048, 4096]):
        super().__init__()

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.bench = bench
        if self.bench == '101':
            self.embedding_lap_pos_enc = nn.Linear(
                pos_enc_dim, d_word_vec)  # position embedding
        elif self.bench == '201':
            self.pos_map = nn.Linear(pos_enc_dim, n_src_vocab + 1)
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, d_word_vec)
            self.proj_func = nn.Linear(4, 6)
        else:
            raise ValueError('No defined NAS bench.')

        # pine structure
        self.conv1 = GATConv(in_features, pine_hidden, heads=heads)
        self.lin1 = torch.nn.Linear(in_features, heads * pine_hidden)

        self.conv2 = GATConv(heads * pine_hidden, pine_hidden, heads=heads)
        self.lin2 = torch.nn.Linear(heads * pine_hidden, heads * pine_hidden)

        self.conv3 = GATConv(
            heads * pine_hidden, linear_input, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(heads * pine_hidden, linear_input)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(
                d_model,
                d_inner,
                n_head,
                d_k,
                d_v,
                dropout=dropout,
                pine_hidden=pine_hidden,
                bench=bench) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # zcp embedder
        self.zcp_embedder_dims = zcp_embedder_dims
        self.zcp_embedder = []
        if bench == '101':
            mid_zcp_dim = 83 * 3 # for nb101
            emb_out_dim = 7 * linear_input # pos_enc_dim
        elif bench == '201':
            mid_zcp_dim = 98 * 3  # for nb201
            emb_out_dim = 4 * linear_input # pos_enc_dim

        for zcp_emb_dim in self.zcp_embedder_dims:  # [128, 128]
            self.zcp_embedder.append(
                nn.Sequential(
                    nn.Linear(mid_zcp_dim, zcp_emb_dim),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=dropout),
                ))
            mid_zcp_dim = zcp_emb_dim
        self.zcp_embedder.append(nn.Linear(mid_zcp_dim, emb_out_dim))
        self.zcp_embedder = nn.Sequential(*self.zcp_embedder)

        self.gate_block = MixerGateBlock(
            emb_out_dim,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.1)

    def to_pyg_batch(self, xs, edge_index_list, num_nodes):
        # import pdb; pdb.set_trace()
        assert xs.shape[0] == len(
            edge_index_list), f'{xs.shape[0]}, {len(edge_index_list)}'
        assert xs.shape[0] == len(
            num_nodes), f'{xs.shape[0]}, {len(num_nodes)}'
        data_list = []
        for x, e, n in zip(xs, edge_index_list, num_nodes):
            data_list.append(torch_geometric.data.Data(x=x[:n], edge_index=e))
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        return batch

    def forward(
            self,
            src_seq,  # features: bs, 7
            pos_seq,  # lapla: bs, 7, 7
            operations,  # operations: bs, 7, 5 -> bs, 35
            edge_index_list,  # list with different length tensor
            num_nodes,  # num of node: bs
            zcp_layerwise,  # zcp: bs, 83
            src_mask=None):
        # op emb and pos emb
        enc_output = self.src_word_emb(src_seq)
        if self.bench == '101':
            pos_output = self.embedding_lap_pos_enc(
                pos_seq)  # positional embedding
            enc_output += pos_output  # bs, 7, 80
            # enc_output = pos_output
            enc_output = self.dropout(enc_output)
        elif self.bench == '201':
            pos_output = self.pos_map(pos_seq).transpose(1, 2)
            pos_output = self.embedding_lap_pos_enc(pos_output)
            enc_output += pos_output
            enc_output = self.dropout(enc_output)
        else:
            raise ValueError('No defined NAS bench.')
        
        # PITE
        x = operations  # bs, 7, 5
        bs = operations.shape[0]  # bs=10 for test
        pyg_batch = self.to_pyg_batch(x, edge_index_list, num_nodes)
        # pyg_batch.x.shape=[70, 5]
        # pyg_batch.edge_index.shape=[2, 84]
        x = F.elu(
            self.conv1(pyg_batch.x, pyg_batch.edge_index) +
            self.lin1(pyg_batch.x))
        x = F.elu(self.conv2(x, pyg_batch.edge_index) + self.lin2(x))
        x = self.conv3(x, pyg_batch.edge_index) + self.lin3(x)
        x = x.view(bs, -1, x.shape[-1])
        if self.bench == '201':
            x = x.transpose(1, 2)
            x = self.proj_func(x)
            x = x.transpose(1, 2)
        enc_output += self.dropout(x)
        enc_output = self.layer_norm(enc_output)

        # zc embedder
        zc_embed = self.zcp_embedder(zcp_layerwise)
        zc_embed = self.gate_block(zc_embed)
        # reshape 
        if self.bench == '101':
            zc_embed = zc_embed.view(bs, 7, -1)
        elif self.bench == '201':
            zc_embed = zc_embed.view(bs, 4, -1)
        
        enc_output += zc_embed

        # backone forward for n_layers (3)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, edge_index_list, num_nodes, slf_attn_mask=src_mask)
        return enc_output




class PINATModel6(nn.Module):
    """
        PINATModel6 is trying to combine zcp into the transformer predictor
        PINATModel5 + zcp embedding (naive embedder)
        PINATModel4 is a small size model with [128,128]
        Here We use large Model
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
                 linear_hidden=512,#80,#80,
                 pine_hidden=256,
                 bench='101'):
        super(PINATModel6, self).__init__()

        # backone
        self.bench = bench
        self.adj_type = adj_type
        self.encoder = Encoder6(
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


    def forward(self, inputs):
        # get arch topology
        numv = inputs['num_vertices']
        assert self.adj_type == 'adj_lapla', f'only support adj_lapla but got {self.adj_type}'
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
            edge_index_list=edge_index_list,
            zcp_layerwise=inputs['zcp_layerwise'])

        # regressor forward
        out = graph_pooling(out, numv)

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out
