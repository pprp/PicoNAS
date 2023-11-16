import torch
import torch.nn as nn
import torch.optim as optim


class BayesianLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(BayesianLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Parameters for the mean and variance
        self.weight_mu = nn.Parameter(torch.Tensor(output_size, input_size))
        self.weight_rho = nn.Parameter(torch.Tensor(output_size, input_size))

        self.register_buffer('weight_eps', torch.Tensor(output_size, input_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight_mu, a=2**0.5)
        nn.init.constant_(self.weight_rho, -3)

    def forward(self, input):
        # Reparameterization trick for the weights
        epsilon = torch.randn_like(self.weight_eps)
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * epsilon

        # Linear transformation with Bayesian weights
        return torch.mm(input, weight.t())


class BayesianNetwork(nn.Module):
    def __init__(self, layer_sizes=[10, 5, 5, 1]):
        super(BayesianNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [
                BayesianLayer(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x['zcp_layerwise']
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


# Example Usage
# layer_sizes = [10, 5, 5, 1]  # Define your layer sizes
# model = BayesianNetwork(layer_sizes)

# # Dummy input (batch_size, number_of_layers)
# x = torch.randn(32, 10)  # Batch size 32, 10 features (layer-wise scores)

# # Forward pass
# output = model(x)
# print(output)
