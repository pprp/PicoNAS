import matplotlib.pyplot as plt
import numpy as np

# Load data from 'gbdt.csv'
data = np.loadtxt('gbdt.csv', delimiter=',', skiprows=1)
feature_importance = data[:, 1]
pos = data[:, 0]

# Create a figure with adjusted colors and styles
plt.figure(figsize=(9, 6))
cmap = plt.get_cmap('viridis')

# Sort the data by feature importance
sorted_idx = np.argsort(feature_importance)
sorted_feature_importance = feature_importance[sorted_idx]
sorted_pos = pos[sorted_idx]

# Use a colormap with more distinguishable colors
colors = cmap(np.linspace(0, 1, len(sorted_feature_importance)))

plt.bar(sorted_pos, sorted_feature_importance, align='center', color=colors)
plt.yticks(fontsize=14)
plt.xlabel('Layer Index', fontsize=18)
plt.ylabel('Relative Importance', fontsize=18)

# Annotate x-axis labels with an interval of 20
x_ticks = np.arange(0, len(sorted_feature_importance) + 1, 20)
plt.xticks(x_ticks, x_ticks, fontsize=14, rotation=45)

plt.yscale('log')

# Add a colorbar for reference
sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(
        vmin=min(feature_importance), vmax=max(feature_importance)))
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical', pad=0.02)

plt.tight_layout()
plt.savefig('gbdt_beautify.png')
plt.show()
