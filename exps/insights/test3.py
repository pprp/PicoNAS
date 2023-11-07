import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

zc_name_list = ['plain_layerwise', 'snip_layerwise',
                'grad_norm_layerwise'] 
#, 'fisher_layerwise', 'l2_norm_layerwise', 'grasp_layerwise']

# Define a list of custom colors suitable for scientific figures with higher opacity
custom_colors = ['#e377c2', '#ff7f0e', '#2ca02c'] 
#, '#d62728', '#9467bd', '#8c564b']

# Create a single figure using "ggplot" style
with plt.style.context('default'):
    plt.figure(figsize=(10, 6))

    # Create an empty list to store legend handles
    legend_handles = []

    # Initialize a dictionary to store the data for sorting and plotting
    data_dict = {}

    for i, zc_name in enumerate(zc_name_list):
        data = np.loadtxt(f'gbdt_{zc_name}.csv', delimiter=',', skiprows=1)
        feature_importance = data[:, 1]
        pos = data[:, 0]

        # Normalize feature importance to enhance discriminability
        scaler = MinMaxScaler()
        feature_importance_normalized = scaler.fit_transform(
            feature_importance.reshape(-1, 1)).flatten()

        # Store data in the dictionary
        data_dict[zc_name] = {
            'pos': pos,
            'feature_importance_normalized': feature_importance_normalized
        }

    # Create corresponding color map for data_dict
    
    