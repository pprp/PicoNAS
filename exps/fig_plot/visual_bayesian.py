import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

# Assuming BayesianLinear and BayesianMLP classes are defined as provided above.

# Initialize the BayesianMLP
input_size = ... # Set the input size
hidden_size = ... # Set the hidden layer size
output_size = ... # Set the output size
bayesian_mlp = BayesianMLP(input_size, hidden_size, output_size)

# Dummy input data for demonstration, replace with your actual data
x = torch.randn((1, input_size))

# Run the network multiple times to get a distribution of outputs
num_samples = 1000  # Number of samples to illustrate uncertainty
outputs = torch.zeros(num_samples, output_size)

for i in range(num_samples):
    outputs[i] = bayesian_mlp(x)

# Now let's plot the distribution of one of the output features
feature_index = 0  # Index of the feature to visualize
feature_outputs = outputs[:, feature_index].detach().numpy()

plt.hist(feature_outputs, bins=50, alpha=0.5)
plt.title('Distribution of Feature {} Outputs'.format(feature_index))
plt.xlabel('Output Value')
plt.ylabel('Frequency')
plt.show()
