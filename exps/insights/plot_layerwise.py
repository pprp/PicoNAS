import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


VISUALIZE = True

# Read file
with open('./data/zc_nasbench201_layerwise.json', 'rb') as f:
    input_dict = json.load(f)
    
ds_target = 'cifar10'  # cifar100, ImageNet16-120
input_dict = input_dict[ds_target]
# plain_layerwise, snip_layerwise, synflow_layerwise grad_norm_layerwise fisher_layerwise l2_norm_layerwise grasp_layerwise
zc_target = 'snip_layerwise'


# Setting up the plot - 3x3 grid for 9 subfigures
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle(f'Layerwise Zero-Cost Scores for {ds_target} using {zc_target}')

for i, (key, value) in enumerate(input_dict.items()):
    # Retrieve the zero-cost scores, replace NaN with 0
    v = value[zc_target]
    v = [0 if np.isnan(x) else x for x in v]
    xs = np.arange(len(v))
    
    # Determine the subplot row and column
    row = i // 3
    col = i % 3

    # Plot subfigure
    axes[row, col].plot(xs, v, label=key)
    axes[row, col].set_title(key)
    axes[row, col].set_xlabel('Layer')
    axes[row, col].set_ylabel('Score')
    axes[row, col].legend()
    axes[row, col].grid(True)
    
    if i == 8:  # Only plot the first 9 subfigures
        break

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show plot
plt.savefig('snip_layerwise_zc_scores.png')