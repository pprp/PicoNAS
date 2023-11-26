import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json 
from tqdm import tqdm

# Architecture data
architecture_data = json.load(open("/data2/dongpeijie/bench/zc_nasbench201_layerwise.json", "r"))['cifar10']

# Use times-new-roman font
# plt.rcParams['font.family'] = 'Times New Roman'

# zc name 
zc_name_list = ['fisher_layerwise', 'grad_norm_layerwise', 'l2_norm_layerwise', 
                'plain_layerwise', 'snip_layerwise', 'synflow_layerwise', 'grasp_layerwise']
zc_color_list = ['steelblue', 'darkorange', 'forestgreen', 'firebrick', 'darkviolet', 'gold', 'black']

plt.figure(figsize=(8, 6))

for idx, zc_name in tqdm(enumerate(zc_name_list)):
    zc_values = {k: v[zc_name] for k, v in architecture_data.items()}

    # filter Nan value to 0
    for k, v in zc_values.items():
        zc_values[k] = [0 if np.isnan(x) else x for x in v]

    # Find the maximum length of zc values
    max_length = max(len(zc) for zc in zc_values.values())

    # Pad the zc values to have the same length
    zc_padded = [zc + [0] * (max_length - len(zc)) for zc in zc_values.values()]

    # Convert the zc values to a numpy array
    zc_array = np.array(zc_padded)


    # Apply t-SNE
    tsne = TSNE(n_components=2)
    embedded_data = tsne.fit_transform(zc_array)


    # Set up the figure

    # Customize marker size and color
    marker_size = 20
    marker_color = zc_color_list[idx]

    # Plot the data
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1], s=marker_size, c=marker_color, alpha=0.1) 

plt.legend(zc_name_list, loc='lower left', fontsize=8)

# Set axis labels and title
plt.xlabel('dim-1')
plt.ylabel('dim-2')
plt.title('t-SNE Visualization')

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)

# Save the figure
plt.savefig(f'tsne_all.png', dpi=300, bbox_inches='tight')