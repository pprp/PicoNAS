import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

cmap = plt.get_cmap('viridis')

# Load data from 'gbdt.csv'
data = np.loadtxt('gbdt.csv', delimiter=',', skiprows=1)
feature_importance = data[:, 1]
pos = data[:, 0]

# Normalize feature importance to enhance discriminability
scaler = MinMaxScaler()
feature_importance_normalized = scaler.fit_transform(
    feature_importance.reshape(-1, 1)
).flatten()

# Apply Mean Filter to feature importance
smooth_filter = np.ones(3) / 3
feature_importance_normalized = np.convolve(
    feature_importance_normalized, smooth_filter, mode='same'
)

# Create a figure with adjusted colors and styles
plt.figure(figsize=(14, 7))
plt.stackplot(
    pos,
    feature_importance_normalized,
    baseline='wiggle',
    colors=cmap(np.linspace(0.2, 0.8, len(pos))),
)

plt.yticks(fontsize=14)
plt.xlabel('Layer Index', fontsize=18)
plt.ylabel('Relative Importance (Normalized)', fontsize=18)
plt.xticks(np.arange(min(pos), max(pos) + 1, 20), fontsize=14, rotation=45)

plt.yscale('log')

# Adjust the colorbar to reflect the normalized data
plt.colorbar(
    plt.cm.ScalarMappable(
        norm=plt.Normalize(
            vmin=min(feature_importance_normalized),
            vmax=max(feature_importance_normalized),
        ),
        cmap=cmap,
    ),
    orientation='vertical',
    pad=0.02,
)

plt.tight_layout()
plt.savefig('./gbdt_streamgraph_normalized.png')
# plt.show()
