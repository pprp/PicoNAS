import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Load data from a CSV file
df = pd.read_csv('./examples/rdnas/visual/predicts_targets.csv'
                 )  # Replace 'your_data.csv' with your CSV file path
x_label = 'ParZC score'
y_label = 'Ground Truth'

# Extract the columns from the DataFrame
data1 = df['predicts'].values
data2 = df['targets'].values

# filter ground truth that is lower than -6
data1 = data1[data2 > -5]
data2 = data2[data2 > -5]

# Calculate the Spearman rank correlation coefficient and p-value
correlation_coefficient, p_value = spearmanr(data1, data2)

# Create a scatter plot with increased transparency (alpha=0.5)
plt.figure(figsize=(8, 6))
plt.scatter(
    data1,
    data2,
    color='blue',
    marker='o',
    s=100,
    alpha=0.5,
    label='Data Points')

# Add labels and title
plt.xlabel(x_label, fontsize=14)
plt.ylabel(y_label, fontsize=14)
plt.title('Spearman Rank Correlation', fontsize=16)

# Add correlation coefficient and p-value to the plot
textstr = f'Spearman Rank Correlation: {correlation_coefficient:.2f}\np-value: {p_value:.4f}'
plt.text(
    0.05,
    0.95,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round, pad=0.5', facecolor='white', alpha=0.8))

# Customize axis ticks and legend
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.yscale('symlog')

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.savefig('Correlation_analytic.png')
