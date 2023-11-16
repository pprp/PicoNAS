import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

zc_name_list = [
    'plain_layerwise',
    'snip_layerwise',
    'grad_norm_layerwise',
    'fisher_layerwise',
    'l2_norm_layerwise',
    'grasp_layerwise',
]

# Define a list of custom colors suitable for scientific figures with higher opacity
custom_colors = ['#e377c2', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Create a single figure using "ggplot" style
with plt.style.context('default'):
    plt.figure(figsize=(12, 4))

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
            feature_importance.reshape(-1, 1)
        ).flatten()

        # Store data in the dictionary
        data_dict[zc_name] = {
            'pos': pos,
            'feature_importance_normalized': feature_importance_normalized,
        }

    # Create corresponding color map for data_dict
    color_map = dict(zip(zc_name_list, custom_colors))

    # Calculate the sorted order of bars based on their values
    sorted_order = np.argsort(
        -data_dict[zc_name_list[0]]['feature_importance_normalized']
    )

    # Plot the bars in the sorted order
    for i, zc_name in enumerate(zc_name_list):
        pos = data_dict[zc_name]['pos'][sorted_order]
        feature_importance_normalized = data_dict[zc_name][
            'feature_importance_normalized'
        ][sorted_order]

        # Plot the bars with custom color and alpha (opacity)
        plt.bar(
            pos,
            feature_importance_normalized,
            align='center',
            color=custom_colors[i],
            alpha=0.5,
            width=0.9,
        )
        # plt.plot(pos, feature_importance_normalized, color=custom_colors[i], alpha=0.7, linewidth=1.5)

        # Collect handles for legend
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=custom_colors[i], alpha=0.5)
        )

    plt.yticks(fontsize=14)
    plt.xlabel('Layer Index', fontsize=16)
    plt.ylabel('Relative Importance', fontsize=16)
    plt.xticks(np.arange(min(pos), max(pos) - 1, 20), fontsize=14, rotation=45)
    # plt.yscale('function', functions=(lambda x: x ** (1 / 3), lambda x: x ** 3))
    plt.yscale('log')

    # Show the legend using collected handles and labels
    plt.legend(
        handles=legend_handles, labels=zc_name_list, loc='upper right', fontsize=12
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./gbdt_streamgraph_combined.png')
    plt.show()
