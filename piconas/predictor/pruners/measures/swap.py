import numpy as np
import torch
import torch.nn as nn
from . import measure 
import pandas as pd

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def cal_regular_factor(model, mu, sigma):

    model_params = torch.as_tensor(count_parameters_in_MB(model)*1e3)
    regular_factor =  torch.exp(-(torch.pow((model_params-mu),2)/sigma))
   
    return regular_factor


class SampleWiseActivationPatterns(object):
    def __init__(self, device):
        self.swap = -1 
        self.activations = None
        self.device = device

    @torch.no_grad()
    def collect_activations(self, activations):
        n_sample = activations.size()[0]
        n_neuron = activations.size()[1]

        if self.activations is None:
            self.activations = torch.zeros(n_sample, n_neuron).to(self.device)  

        self.activations = torch.sign(activations)

    @torch.no_grad()
    def calSWAP(self, regular_factor):
        
        self.activations = self.activations.T # transpose the activation matrix: (samples, neurons) to (neurons, samples)
        self.swap = torch.unique(self.activations, dim=0).size(0)
        
        del self.activations
        self.activations = None
        torch.cuda.empty_cache()

        return self.swap * regular_factor


class SWAP:
    def __init__(self, model=None, inputs = None, device='cpu', seed=0, regular=True, mu=None, sigma=None):
        self.model = model
        self.interFeature = []
        self.seed = seed
        self.regular = regular
        self.regular_factor = 1
        self.mu = mu
        self.sigma = sigma
        self.inputs = inputs
        self.device = device
        self.reinit(self.model, self.seed)

    def reinit(self, model=None, seed=None):
        if model is not None:
            self.model = model
            self.register_hook(self.model)
            self.swap = SampleWiseActivationPatterns(self.device)
            if self.regular and self.mu is not None and self.sigma is not None:
                self.regular_factor = cal_regular_factor(self.model, self.mu, self.sigma).item()
 
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.swap = SampleWiseActivationPatterns(self.device)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for n, m in model.named_modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach()) 

    def forward(self):
        self.interFeature = []
        with torch.no_grad():
            self.model.forward(self.inputs.to(self.device))
            if len(self.interFeature) == 0: return
            activtions = torch.cat([f.view(self.inputs.size(0), -1) for f in self.interFeature], 1)         
            self.swap.collect_activations(activtions)
            
            return self.swap.calSWAP(self.regular_factor) 
        

@measure('swap', bn=True)
def compute_swap(net, inputs, targets, split_data=1, loss_fn=None, seed=0, regular=True, mu=None, sigma=None):
    model_params = count_parameters_in_MB(net) * 1e3
    model_params = pd.Series(model_params)
    # Automatically set mu and sigma, also can be set manually
    mu = model_params.describe()[7]
    sigma = model_params.describe()[1]

    swap = SWAP(model=net, inputs=inputs, device=inputs.device, seed=seed, regular=regular, mu=mu, sigma=sigma)

    return swap.forward()