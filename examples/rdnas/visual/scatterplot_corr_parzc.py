import matplotlib.pyplot as plt
import numpy as np 
import csv 
from piconas.utils.rank_consistency import kendalltau 
from scipy.stats import gaussian_kde

plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['font.size'] = 12

# set dpi 
plt.rcParams['savefig.dpi'] = 300

# read file from correlation_parzc.csv have two columns: predicts and targets
f = open('correlation_parzc.csv', 'r')
reader = csv.reader(f)
predicts = []
targets = []
for i, row in enumerate(reader):
    if i == 0:
        continue
    predicts.append(float(row[0]))
    targets.append(float(row[1]))

# convert to numpy array
predicts = np.array(predicts)
targets = np.array(targets)

# calculate density
xy = np.vstack([predicts, targets])
z = gaussian_kde(xy)(xy)

idx = z.argsort()
predicts, targets, z = predicts[idx], targets[idx], z[idx]

# filter out architectures gt < -5 
z = z[targets > -5]
predicts = predicts[targets > -5]
targets = targets[targets > -5]

plt.scatter(
    predicts,
    targets,
    alpha=0.3,
    c=z,
    s=5,
    label='kendall_tau: %.4f' % kendalltau(predicts, targets))

# Label and title
plt.xlabel('Predicted Performance')
plt.ylabel('Ground Truth Performance')

# Adjust axis limits
plt.xlim(min(predicts), max(predicts))
plt.ylim(min(targets), max(targets))

# Add a legend
plt.legend()

# Save the figure
plt.savefig('scatterplot.png')
plt.close() 

# filter the top 10% architectures
top_idx = np.argsort(predicts)[-int(len(predicts) * 0.05):]
predicts = predicts[top_idx]
targets = targets[top_idx]

# calculate density
xy = np.vstack([predicts, targets])
z = gaussian_kde(xy)(xy)

idx = z.argsort()
predicts, targets, z = predicts[idx], targets[idx], z[idx]

plt.scatter(
    predicts,
    targets,
    alpha=0.3,
    c=z,
    s=5)

# show the top-3 with highest targets with star 
top_idx = np.argsort(targets)[-3:]
plt.scatter(
    predicts[top_idx],
    targets[top_idx],
    alpha=0.8,
    c='r',
    s=50,
    marker='*',
    label='top-3 arch')


# Label and title
plt.xlabel('Predicted Performance')
plt.ylabel('Ground Truth Performance')

# Adjust axis limits
plt.xlim(min(predicts), max(predicts)*1.01)
plt.ylim(min(targets), max(targets)*1.02)

# Add a legend
plt.legend()

plt.tight_layout()
# Save the figure
plt.savefig('scatterplot_top.png')
