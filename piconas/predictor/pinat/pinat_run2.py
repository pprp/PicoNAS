# Run2: only zcp

import torch.nn as nn


class PINATModel2(nn.Module):
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
        mid_zcp_dim = 13
        for zcp_emb_dim in self.zcp_embedder_dims:
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
        if hasattr(inputs, 'zcp') and inputs['zcp'] is not None:
            zc_embed = self.zcp_embedder(inputs['zcp'])
            out = zc_embed
        else:
            raise NotImplementedError('No zcp input.')

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out).view(-1)
        return out
