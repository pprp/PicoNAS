
from piconas.predictor.pinat.model_factory import create_best_nb201_model
import torch 
from piconas.datasets.predictor.nb201_dataset import Nb201DatasetPINAT
import numpy as np 

# build predictor model 
predictor_m = create_best_nb201_model() 
ckpt_dir = 'checkpoints/nasbench_201/201_cifar10_ParZCBMM_mse_t781_vall_e153_bs10_best_nb201_run2_tau0.783145_ckpt.pt'
predictor_m.load_state_dict(
    torch.load(ckpt_dir, map_location=torch.device('cpu')))


# build dataset for nb201. Test set encompass all the architectures in the search space
test_set = Nb201DatasetPINAT(
    split='all', data_type='test', data_set='cifar10')

# refer to rdnas_trainer.py line 554 
ss_index = 1 
input = test_set.get_batch(ss_index)

key_list = [
    'num_vertices', 'lapla', 'edge_num', 'features', 'zcp_layerwise'
]
input['edge_index_list'] = [input['edge_index_list']]
input['operations'] = torch.tensor(input['operations']).unsqueeze(0).unsqueeze(0) 

for _key in key_list:
    if isinstance(input[_key], (list, float, int)):
        input[_key] = torch.tensor(input[_key])
        input[_key] = torch.unsqueeze(
            input[_key], dim=0)
    elif isinstance(input[_key], np.ndarray):
        input[_key] = torch.from_numpy(input[_key])
        input[_key] = torch.unsqueeze(
            input[_key], dim=0)
    elif isinstance(input[_key], torch.Tensor):
        input[_key] = torch.unsqueeze(
            input[_key], dim=0)
    else:
        raise NotImplementedError(
            f'key: {_key} is not list, is a {type(input[_key])}')
    input[_key] = input[_key]
score = predictor_m(input)

print(score.item())