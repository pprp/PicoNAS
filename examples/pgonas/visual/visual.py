import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-whitegrid')

spos_csv = './spos_baseline.csv'
ours_csv = './ours.csv'

x = []
y_spos = []
y_ours = []
choice = 13
max_second = 40000
x_list = np.linspace(0, 40000, 21)

with open(spos_csv, 'r') as f:
    spos_contents = csv.reader(f)
    for i, c in enumerate(spos_contents):
        if i == 0:
            continue
        y_spos.append(float(c[2]))

with open(ours_csv, 'r') as f:
    ours_contents = csv.reader(f)
    for i, c in enumerate(ours_contents):
        if i == 0:
            continue
        y_ours.append(float(c[2]))
        x.append(x_list[i - 1])


def smooth_list(value_list: list):
    new_list = []
    for i, v in enumerate(value_list):
        if i > 0 and i < len(value_list) - 2:
            new_list.append(
                (value_list[i + 1] + value_list[i - 1] + value_list[i]) / 3)
        else:
            new_list.append(value_list[i])
    return new_list


# zero-cost proxies.
# random, flops, params, snip, synflow, zennas, nwot
zc_proxies = [-0.26, 0.5292, 0.5664, 0.5116, 0.5883, 0.4880, 0.5513, 0.3405]
zc_label = [
    'random', 'flops', 'params', 'snip', 'synflow', 'zennas', 'nwot', 'grasp'
]
zc_marker = ['.', ',', 'o', 'v', '^', '<', '>', 'o']
zc_color = ['g', 'b', 'r', 'y', 'k', 'm', 'g', 'b']

y_spos = smooth_list(y_spos)
y_ours = smooth_list(y_ours)

if choice is not None:
    x = x[:choice]
    y_spos = y_spos[:choice]
    y_ours = y_ours[:choice]

plt.figure(figsize=(12, 8), dpi=100)
plt.grid(True, linestyle='--', alpha=0.8)

plt.plot(
    x,
    y_spos,
    color='g',
    mfc='white',
    linewidth=2,
    marker='^',
    linestyle=':',
    label='spos')
plt.plot(
    x,
    y_ours,
    color='r',
    mfc='white',
    linewidth=2,
    marker='o',
    linestyle='-',
    label='rd-nas')

# zc
for i, zc in enumerate(zc_proxies):
    plt.scatter(
        0,
        zc,
        c=zc_color[i],
        s=50,
        cmap='viridis',
        marker=zc_marker[i],
        label=zc_label[i])

plt.yticks(fontproperties='Times New Roman', size=20)
plt.xticks(fontproperties='Times New Roman', size=20)
plt.xlim(-1000, 25000)
plt.ylim(-0.3, 0.8)

plt.legend()
plt.xlabel('Time (sec)', fontdict={'family': 'Times New Roman', 'size': 18})
plt.ylabel('Kendall tau', fontdict={'family': 'Times New Roman', 'size': 18})
plt.show()
