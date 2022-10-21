import csv

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')

spos_csv = './spos_baseline.csv'
ours_csv = './ours.csv'

x = []
y_spos = []
y_ours = []
# x_time_label = ['0', '5,000', '10,000', '15,000', '20,000', '25,000', '30,000',
#                 '35,000', '40,000', '45,000', '50,000', '55,000']
x_time_label = [
    '0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55'
]

from_idx = 0
choice = 12
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
    'Random', 'FLOPs', 'Params', 'Snip', 'Synflow', 'ZenNAS', 'Nwot', 'Grasp'
]
zc_marker = ['X', ',', 'o', 'v', 'D', 'p', '>', '^']
zc_color = ['mediumblue', 'b', 'k', 'y', 'r', 'orange', 'g', 'grey']

y_spos = smooth_list(y_spos)
y_ours = smooth_list(y_ours)

if choice is not None:
    x = x[from_idx:choice]
    y_spos = y_spos[from_idx:choice]
    y_ours = y_ours[from_idx:choice]
    x_time_label = x_time_label[from_idx:]

plt.figure(figsize=(12, 9), dpi=100)
plt.grid(True, linestyle='-', alpha=0.9)

plt.plot(
    x,
    y_spos,
    color='g',
    # mfc='white',
    linewidth=4,
    marker='o',
    linestyle=':',
    label='SPOS',
    markersize=12)
plt.plot(
    x,
    y_ours,
    color='r',
    # mfc='white',
    linewidth=4,
    marker='v',
    markersize=12,
    label='RD-NAS')

# zc
for i, zc in enumerate(zc_proxies):
    plt.scatter(
        0,
        zc,
        c=zc_color[i],
        s=80,
        cmap='viridis',
        marker=zc_marker[i],
        label=zc_label[i])

plt.title(
    'Rank Consistency of NAS-Bench-201 CIFAR10',
    fontproperties='Times New Roman',
    size=33)
plt.xlim(-1000, 22500)
plt.ylim(-0.35, 0.8)
# plt.ylim(0, 0.8)
plt.yticks(np.linspace(0, 0.8, 5), fontproperties='Times New Roman', size=27)
plt.xticks(x, x_time_label, fontproperties='Times New Roman', size=27)

plt.annotate(
    'Ours',
    xy=(5000, 0.63),
    fontsize=28,
    c='r',
    fontproperties='Times New Roman')
# plt.annotate('100% faster', xy=(6000, 0.6), xytext=(12000, 0.6),
#              arrowprops=dict(facecolor='r', shrink=0.01, ec='r'), fontsize=28, c='r')

plt.legend(fontsize=22, loc=4)
plt.xlabel(
    '$10^3$Time (seconds)', fontdict={
        'family': 'Times New Roman',
        'size': 33
    })
plt.ylabel("Kendall's Tau", fontdict={'family': 'Times New Roman', 'size': 33})
plt.show()
