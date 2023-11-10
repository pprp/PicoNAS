from piconas.predictor.pinat.bayesian import BayesianNetwork
from piconas.predictor.pinat.BN.bayesian import BayesianNetwork
from piconas.predictor.pinat.gcn import NeuralPredictorModel
from piconas.predictor.pinat.mlpmixer import MLPMixer
from piconas.predictor.pinat.pinat_model import (ParZCBMM, ParZCBMM2,
                                                 PINATModel1, PINATModel2,
                                                 PINATModel3, PINATModel4,
                                                 PINATModel5, PINATModel6,
                                                 PINATModel7)

_name2model = {
    'PINATModel1': PINATModel1,  # PINAT + ZCP
    'PINATModel2': PINATModel2,  # ZCP only
    'PINATModel3': PINATModel3,  # PINAT + ZCP + BN
    'PINATModel4': PINATModel4,  # PINAT + ZCP Layerwise + Gating
    'PINATModel5':
    PINATModel5,  # PINAT + ZCP Layerwise + Gating + Larger Model
    'PINATModel6':
    PINATModel6,  # PINAT + ZCP Layerwise + Gating + Larger Model Modify Encoder
    'PINATModel7':
    PINATModel7,  # PINAT + ZCP Layerwise + Gating + Larger Model Modify Encoder + bayesian network
    'ParZCBMM': ParZCBMM,  # ZCP + BMM
    'ParZCBMM2': ParZCBMM2,
    # ablation study
    'BayesianNetwork': BayesianNetwork,
    'MLPMixer': MLPMixer,
    'NeuralPredictorModel': NeuralPredictorModel
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
        d_word_vec=512,  # 80
        d_k=64,
        d_v=64,
        d_model=512,  # 80
        d_inner=512,
    )

    return net


def create_best_nb201_model():
    pos_enc_dim_dict = {'101': 7, '201': 4}
    net = ParZCBMM(
        bench='201',
        pos_enc_dim=pos_enc_dim_dict['201'],
        adj_type='adj_lapla',
        n_layers=4,
        n_head=6,
        pine_hidden=76,
        linear_hidden=765,
        n_src_vocab=5,
        d_word_vec=765,  # 80
        d_k=100,
        d_v=100,
        d_model=765,  # 80
        d_inner=338)
    return net


def create_best_nb101_model():
    pos_enc_dim_dict = {'101': 7, '201': 4}
    net = ParZCBMM(
        bench='101',
        pos_enc_dim=pos_enc_dim_dict['101'],
        adj_type='adj_lapla',
        n_layers=4,
        n_head=6,
        pine_hidden=8,
        linear_hidden=708,
        n_src_vocab=5,
        d_word_vec=708,  # 80
        d_k=100,
        d_v=100,
        d_model=708,  # 80
        d_inner=530)
    return net


def create_model_hpo(n_layers, n_head, pine_hidden, linear_hidden,
                     d_word_model, d_k_v, d_inner):
    pos_enc_dim_dict = {'101': 7, '201': 4}
    net = ParZCBMM(
        bench='101',
        pos_enc_dim=pos_enc_dim_dict['101'],
        adj_type='adj_lapla',
        n_layers=n_layers,
        n_head=n_head,
        pine_hidden=pine_hidden,
        linear_hidden=linear_hidden,
        n_src_vocab=5,
        d_word_vec=d_word_model,  # 80
        d_k=d_k_v,
        d_v=d_k_v,
        d_model=d_word_model,  # 80
        d_inner=d_inner)
    return net


def create_nb201_model():
    pos_enc_dim_dict = {'101': 7, '201': 4}
    net = PINATModel7(
        bench='201',
        pos_enc_dim=pos_enc_dim_dict['201'],
        adj_type='adj_lapla',
        n_layers=3,
        n_head=4,
        pine_hidden=16,
        linear_hidden=96,
        n_src_vocab=5,
        d_word_vec=512,  # 80
        d_k=64,
        d_v=64,
        d_model=512,  # 80
        d_inner=512,
    )

    return net
