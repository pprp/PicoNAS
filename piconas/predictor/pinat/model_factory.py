from piconas.predictor.pinat.pinat_model import (PINATModel1, PINATModel2,
                                                 PINATModel3, PINATModel4)

_name2model = {
    'PINATModel1': PINATModel1,  # PINAT + ZCP
    'PINATModel2': PINATModel2,  # ZCP only
    'PINATModel3': PINATModel3,  # PINAT + ZCP + BN
    'PINATModel4': PINATModel4,  # PINAT + ZCP Layerwise + Gating
}


def create_model(args):
    pos_enc_dim_dict = {'101': 7, '201': 4}
    MODEL = _name2model[args.model_name]
    net = MODEL(
        bench=args.bench,
        pos_enc_dim=pos_enc_dim_dict[args.bench],
        adj_type='adj_lapla',
        n_layers=3,
        n_head=4,
        pine_hidden=16,
        linear_hidden=96,
        n_src_vocab=5,
        d_word_vec=80,
        d_k=64,
        d_v=64,
        d_model=80,
        d_inner=512,
    )

    return net


def create_nb201_model():
    pos_enc_dim_dict = {'101': 7, '201': 4}
    net = PINATModel4(
        bench='201',
        pos_enc_dim=pos_enc_dim_dict['201'],
        adj_type='adj_lapla',
        n_layers=3,
        n_head=4,
        pine_hidden=16,
        linear_hidden=96,
        n_src_vocab=5,
        d_word_vec=80,
        d_k=64,
        d_v=64,
        d_model=80,
        d_inner=512,
    )

    return net
