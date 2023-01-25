import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker

plt.rc('font', family='Times New Roman')
GLOBAL_DPI = 600
FIGSIZE = (8, 6)
PADINCHES = 0.1  # -0.005
GLOBAL_FONTSIZE = 34
GLOBAL_LABELSIZE = 30
GLOBAL_LEGENDSIZE = 20

font1 = {
    'family': 'Times New Roman',
    'weight': 'bold',
    'size': GLOBAL_LABELSIZE
}

plt.rc('font', **font1)  # controls default text sizes
plt.rc('axes', titlesize=GLOBAL_LABELSIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=GLOBAL_LABELSIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=GLOBAL_LABELSIZE - 10)  # fontsize of the tick labels
plt.rc('ytick', labelsize=GLOBAL_LABELSIZE - 10)  # fontsize of the tick labels
plt.rc('legend', fontsize=GLOBAL_LEGENDSIZE)  # legend fontsize
plt.rc('figure', titlesize=GLOBAL_LABELSIZE)

ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)


def plot_standalone_model_rank(xlabel, ylabel, foldername, file_name):
    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ticks = np.arange(0, 51, 10)
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks)
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks)
    axs[2].set_xticks(ticks)
    axs[2].set_xticklabels(ticks)
    axs[0].set_xticks(ticks)
    axs[0].set_yticklabels(ticks)
    axs[1].set_yticks(ticks)
    axs[1].set_yticklabels(ticks)
    axs[2].set_yticks(ticks)
    axs[2].set_yticklabels(ticks)
    for ax in axs.flat:
        ax.label_outer()

    ax1 = axs[0]
    real_rank = [
        0, 16, 32, 48, 4, 20, 36, 52, 8, 24, 1, 40, 12, 56, 17, 28, 33, 44, 21,
        5, 60, 49, 37, 2, 53, 25, 9, 50, 13, 18, 3, 34, 57, 29, 61, 41, 6, 45,
        10, 22, 35, 38, 19, 26, 7, 58, 54, 46, 39, 14, 30, 51, 23, 62, 42, 55,
        11, 15, 59, 31, 27, 63, 47, 43
    ]
    angle_rank = [
        28, 0, 12, 36, 32, 16, 24, 20, 8, 60, 4, 52, 40, 48, 44, 56, 9, 5, 21,
        17, 13, 1, 29, 25, 61, 37, 53, 41, 33, 57, 49, 30, 45, 10, 2, 6, 18,
        14, 46, 26, 7, 38, 34, 22, 35, 27, 47, 23, 42, 19, 11, 3, 39, 54, 50,
        15, 31, 58, 51, 59, 63, 62, 55, 43
    ]

    ax1.scatter(real_rank, angle_rank, alpha=0.6)
    ax1.set_title('CIFAR-10')
    ax1.text(real_rank[-1] - 23, 2, 'Tau=0.710', fontsize=20)
    ax1.set_xticks(np.arange(0, real_rank[-1] + 1, 10))
    ax1.set_yticks(np.arange(0, real_rank[-1] + 1, 10))

    ax2 = axs[1]

    real_rank = [
        31, 38, 3, 15, 47, 30, 0, 37, 44, 9, 36, 19, 10, 16, 45, 13, 35, 41,
        25, 1, 28, 14, 22, 18, 48, 46, 42, 33, 32, 26, 40, 34, 24, 23, 5, 12,
        6, 17, 4, 20, 7, 11, 8, 43, 29, 39, 49, 27, 2, 21
    ]
    angle_rank = [
        31, 15, 9, 25, 30, 35, 14, 38, 13, 10, 16, 1, 3, 22, 40, 32, 47, 37,
        46, 48, 11, 5, 2, 42, 45, 0, 23, 34, 18, 8, 26, 6, 12, 33, 41, 36, 19,
        17, 4, 29, 21, 44, 27, 43, 20, 7, 39, 24, 28, 49
    ]

    ax2.scatter(real_rank, angle_rank, alpha=0.6)
    ax2.set_title('CIFAR-100')
    ax2.text(real_rank[-1] - 23, 2, 'Tau=0.471', fontsize=20)
    ax2.set_xticks(np.arange(0, real_rank[-1] + 1, 10))
    ax2.set_yticks(np.arange(0, real_rank[-1] + 1, 10))

    ax3 = axs[2]
    real_rank = [
        28, 1, 22, 37, 39, 19, 27, 25, 33, 8, 29, 17, 24, 49, 21, 20, 44, 4, 5,
        6, 40, 16, 12, 45, 9, 18, 36, 23, 38, 15, 46, 43, 41, 0, 10, 35, 3, 11,
        2, 26, 32, 13, 14, 34, 7, 30, 31, 48, 42, 47
    ]
    angle_rank = [
        17, 1, 28, 29, 25, 22, 27, 35, 39, 20, 19, 38, 11, 44, 0, 15, 8, 12, 3,
        37, 32, 9, 36, 41, 46, 47, 2, 30, 23, 13, 18, 24, 7, 14, 49, 33, 40, 4,
        21, 48, 31, 34, 26, 10, 16, 6, 5, 45, 42, 43
    ]
    ax3.scatter(real_rank, angle_rank, alpha=0.6)
    ax3.set_title('ImageNet-16-120')  # 0.851
    ax3.text(real_rank[-1] - 23, 2, 'Tau=0.324', fontsize=20)
    ax3.set_xticks(np.arange(0, real_rank[-1] + 1, 10))
    ax3.set_yticks(np.arange(0, real_rank[-1] + 1, 10))
    fig.tight_layout()

    save_path = os.path.join(foldername, file_name)
    # foldername / '{}'.format(file_name)
    print('save figure into {:}\n'.format(save_path))
    plt.savefig(
        str(save_path),
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_ranking_stability(xlabel, ylabel, foldername, file_name):
    plt.figure(figsize=FIGSIZE)

    ickd_data = []
    diswot_data = []
    sp_data = []
    datasets = ['cifar10', 'cifar100', 'ImageNet16-120']

    for dataset in datasets:
        if dataset == 'cifar10':
            # random_data.append([random.random() for _ in range(10)])
            # acc_no_rebn_data.append([random.random() for _ in range(10)])
            ickd_data.append([
                0.2434, 0.5166, 0.4982, 0.3267, 0.4508, 0.4443, 0.4303, 0.4067,
                0.4822, 0.5768
            ])
            diswot_data.append([
                0.4802, 0.4044, 0.4315, 0.4877, 0.4142, 0.5253, 0.5304, 0.3731,
                0.5994, 0.4575
            ])
            # kendall's tau
            sp_data.append([
                0.5264, 0.3642, 0.3381, 0.4338, 0.5172, 0.4232, 0.5474, 0.3861,
                0.2098, 0.4706
            ])

    width = 0.20
    locations = list(range(len(ickd_data)))
    locations = [i + 1 - 0.135 for i in locations]

    positions1 = locations
    boxplot1 = plt.boxplot(
        diswot_data,
        positions=positions1,
        patch_artist=True,
        showfliers=True,
        widths=width)

    positions2 = [x + (width + 0.08) for x in locations]
    boxplot2 = plt.boxplot(
        ickd_data,
        positions=positions2,
        patch_artist=True,
        showfliers=True,
        widths=width)

    positions3 = [x + (width + 0.08) * 2 for x in locations]
    boxplot3 = plt.boxplot(
        sp_data,
        positions=positions3,
        patch_artist=True,
        showfliers=True,
        widths=width)

    for box in boxplot1['boxes']:
        box.set(color='#3c73a8')

    for box in boxplot2['boxes']:
        box.set(color='#fec615')

    for box in boxplot3['boxes']:
        box.set(color='#2ec615')

    plt.xlim(0, len(ickd_data) + 1)

    ticks = np.arange(0, len(ickd_data) + 1, 1)
    ticks_label_ = ['CIFAR-10', 'CIFAR-100', 'ImageNet-16-120', '']
    ticks_label = []

    for i in range(len(ickd_data) + 1):
        ticks_label.append(str(ticks_label_[i - 1]))
    plt.xticks(ticks, ticks_label)  # , rotation=45)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)

    plt.grid(lw=2, ls='-.')
    plt.plot([], c='#3c73a8', label='Acc. w/ ReBN')
    plt.plot([], c='#fec615', label='Angle')
    plt.legend(ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.55))

    save_path = os.path.join(foldername, file_name)

    print('save figure into {:}\n'.format(save_path))
    plt.savefig(
        str(save_path),
        bbox_inches='tight',
        dpi=GLOBAL_DPI,
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_evolution_search_process():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    x = []
    y = []
    with open('./exps/evo_search_img.log', 'r') as f:
        contents = f.readlines()
        for i, c in enumerate(contents):
            max_score = float(c.split(', ')[1].split('=')[1])
            x.append(i)
            y.append(max_score)

    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.grid(linestyle='-.', lw=2, alpha=0.9)
    # print(x, y)
    plt.plot(
        x, y, color='salmon', linestyle='-', lw=3, label='Evolution search')

    x2, y2 = [], []
    with open('./exps/rand_search_img.log', 'r') as f:
        contents = f.readlines()
        for i, c in enumerate(contents):
            max_score = float(c.split(', ')[1].split('=')[1])
            x2.append(i)
            y2.append(max_score)

    plt.plot(
        x2, y2, color='skyblue', linestyle='-', lw=4, label='Random search')

    plt.legend()
    xticks = np.arange(0, 1000, 250)
    plt.ylim([-0.00053, -0.00018])
    plt.xticks(xticks, fontsize=GLOBAL_LABELSIZE)
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE)
    plt.xlabel('Iteration', fontsize=GLOBAL_LEGENDSIZE, weight='bold')
    plt.ylabel('DisWOT Metric', font1)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(
        './tmp/ES_vs_RS.pdf',
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_kd_zc_box(foldername, file_name):
    import json
    plt.figure(figsize=FIGSIZE)

    # load from files
    kd_file_path = './exps/kd_name_results_x10.txt'
    zc_file_path = './exps/zc_name_results_x10.txt'
    with open(kd_file_path, 'r') as f:
        kd_info = json.load(f)
    with open(zc_file_path, 'r') as f:
        zc_info = json.load(f)
    # merge kd and zc
    merged_info = {**kd_info, **zc_info}

    total_boxes = len(merged_info)

    plt.figure(figsize=(32, 4))

    width = 0.20
    locations = list(range(total_boxes))
    locations = [i + 1 - 0.135 for i in locations]

    labels = []
    content_list = []
    for i, (k, v) in enumerate(merged_info.items()):
        labels.append(k)
        content_list.append(v)

    plt.boxplot(
        content_list,
        medianprops={
            'color': 'red',
            'linewidth': '1.5'
        },
        meanline=True,
        showmeans=True,
        meanprops={
            'color': 'blue',
            'ls': '-.',
            'linewidth': '1.5'
        },
        flierprops={
            'marker': 'o',
            'markerfacecolor': 'red',
            'markersize': 10
        },
        labels=labels)

    plt.xlim(0, total_boxes + 1)

    plt.grid(lw=2, ls='-.')
    plt.tight_layout()

    save_path = os.path.join(foldername, file_name)

    print('save figure into {:}\n'.format(save_path))
    plt.savefig(
        str(save_path),
        bbox_inches='tight',
        dpi=GLOBAL_DPI,
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_kd_box(foldername, file_name):
    import json
    plt.figure(figsize=FIGSIZE)
    # load from files
    kd_file_path = './exps/kd_name_results_x10.txt'
    with open(kd_file_path, 'r') as f:
        kd_info = json.load(f)

    for k, v in kd_info.items():
        print(k, np.mean(np.array(v)))

    # merge kd and zc
    merged_info = {**kd_info}

    total_boxes = len(merged_info)
    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    locations = list(range(total_boxes))
    locations = [i + 1 - 0.135 for i in locations]

    labels = []
    content_list = []
    for i, (k, v) in enumerate(merged_info.items()):
        labels.append(k)
        content_list.append(v)

    plt.boxplot(
        content_list,
        medianprops={
            'color': 'red',
            'linewidth': '2'
        },
        meanline=True,
        showmeans=True,
        meanprops={
            'color': 'blue',
            'ls': '-.',
            'linewidth': '2'
        },
        flierprops={
            'marker': 'o',
            'markerfacecolor': 'red',
            'markersize': 20
        },
        labels=labels,
        boxprops={'linewidth': '2'},
        whiskerprops={'linewidth': '2'},
        widths=0.8)

    plt.xlim(0, total_boxes + 1)

    plt.grid(lw=2, ls='-.')
    plt.tight_layout()

    save_path = os.path.join(foldername, file_name)
    plt.ylabel(
        'Spearman Coefficienct', fontsize=GLOBAL_LABELSIZE, weight='bold')
    plt.xlabel(
        'Distillation Method', fontsize=GLOBAL_LEGENDSIZE, weight='bold')

    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE - 4)

    print('save figure into {:}\n'.format(save_path))
    plt.savefig(
        str(save_path),
        bbox_inches='tight',
        dpi=GLOBAL_DPI,
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_different_teacher_diswot():
    plt.figure(figsize=FIGSIZE)
    labels = ['ResNet32', 'ResNet44', 'ResNet56', 'ResNet110']
    res20_list = [70.24, 70.56, 70.98, 70.79]
    diswot_list = [71.01, 71.25, 71.63, 71.84]
    diswot_plus_list = [71.85, 72.12, 72.56, 72.92]
    marker_size = 13

    ax = plt.gca()
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.grid(linestyle='-.', alpha=0.9, lw=2)
    plt.plot(
        labels,
        res20_list,
        color='#FFBE7A',
        # mfc='white',
        linewidth=2,
        marker='o',
        # linestyle=':',
        label='KD',
        markersize=marker_size)
    plt.plot(
        labels,
        diswot_list,
        color='#2878B5',
        # mfc='white',
        linewidth=2,
        marker='v',
        markersize=marker_size,
        label='DisWOT')
    plt.plot(
        diswot_plus_list,
        color='#32B897',
        # mfc='white',
        linewidth=2,
        marker='*',
        markersize=marker_size + 5,
        label='DisWOT+')
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE)
    plt.xlabel('Teacher Models', fontsize=GLOBAL_LEGENDSIZE, weight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=GLOBAL_FONTSIZE, weight='bold')
    label_size = plt.legend().get_texts()
    [label.set_fontsize(GLOBAL_LEGENDSIZE - 6) for label in label_size]
    plt.tight_layout()
    plt.savefig(
        './tmp/diff_teacher_diswot.pdf',
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_hist_rank_consistency():
    result = {
        'Fisher': {
            'cls': 0.81,
            'kd': 0.63
        },
        'NWOT': {
            'cls': 0.40,
            'kd': 0.32
        },
        'SNIP': {
            'cls': 0.85,
            'kd': 0.67
        },
        'Vanilla acc.': {
            'cls': 1.00,
            'kd': 0.85
        }
    }

    labels = result.keys()
    x = np.arange(len(labels))
    y1 = [v['cls'] for k, v in result.items()]
    y2 = [v['kd'] for k, v in result.items()]
    width = 0.3

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    rects1 = ax.bar(
        x - width / 2, y1, width, label='Vanilla acc.', color='salmon')
    rects2 = ax.bar(
        x + width / 2, y2, width, label='Distill acc.', color='skyblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Kendall's Tau", fontsize=GLOBAL_FONTSIZE, weight='bold')
    ax.set_xticks(x, labels)  #, fontsize=GLOBAL_LEGENDSIZE - 2)
    yticks = np.arange(0.1, 1.2, 0.1)
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE - 2)
    ax.set_yticks(yticks)  #, fontsize=GLOBAL_LABELSIZE)
    ax.set_ylim(0.2, 1.05)
    label_size = ax.legend().get_texts()
    [label.set_fontsize(GLOBAL_LEGENDSIZE - 3) for label in label_size]

    ax.bar_label(
        rects1, padding=-26, label_type='edge', fontsize=GLOBAL_LEGENDSIZE - 2)
    ax.bar_label(
        rects2, padding=-26, label_type='edge', fontsize=GLOBAL_LEGENDSIZE - 2)

    fig.tight_layout()
    plt.grid(linestyle='-.', alpha=0.9, lw=2)

    plt.savefig(
        './hist_rank_consistency.pdf',
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_param_cls_kd_ours_all():
    import json
    kd_dict = {}
    pre_dict = {}
    with open('./exps/diswot_sp_score_info.txt', 'r') as f:
        js = f.read()
        info_dict = json.loads(js)
        for k, v in info_dict.items():
            kd_dict[k] = v['gt']
            pre_dict[k] = v['pre']

    cls_dict = {}
    with open('./exps/s1-gt-cls.txt', 'r') as f:
        contents = f.readlines()
        for c in contents:
            k, v = c.split()
            cls_dict[k] = float(v)

    bar_width = 0.25

    merged_dict = {}
    for k, kd_acc in kd_dict.items():
        merged_dict[k] = [kd_acc, cls_dict[k], pre_dict[k]]

    labels = merged_dict.keys()
    x = np.arange(len(labels))
    kd_list = [v[0] for k, v in merged_dict.items()]
    cls_list = [v[1] for k, v in merged_dict.items()]
    pre_list = [0.001 - v[2] for k, v in merged_dict.items()]

    fig, ax1 = plt.subplots(1, 1, figsize=(30, 5))
    ax1.bar(x, cls_list, width=bar_width, label='Cls. ACC', fc='steelblue')
    ax1.bar(
        x + bar_width,
        kd_list,
        width=bar_width,
        label='KD. ACC',
        fc='seagreen')
    ax1.set_ylim([55, 80])

    ax2 = ax1.twinx()
    ax2.bar(
        x + bar_width * 2,
        pre_list,
        width=bar_width,
        label='DisWOT',
        fc='indianred')
    # ax2.set_ylim([])
    fig.legend(loc=1)
    ax1.set_xticks(x, labels)
    plt.savefig(
        './tmp/param_cls_kd_diswot.pdf',
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_param_cls_kd_ours():
    bar_width = 0.2

    labels = ['ResNet[713](259.89k)', 'ResNet[333](278.32k)']

    x = np.arange(len(labels))
    kd_list = [71.01, 70.76]
    cls_list = [69.13, 69.57]
    pre_list = [0.002 - 0.0015596, 0.002 - 0.0016668]
    pre_label = ['4.4e-4', '3.3e-4']

    fig, ax1 = plt.subplots(1, 1, figsize=FIGSIZE)

    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    ax1.yaxis.label.set_size(GLOBAL_LEGENDSIZE)

    rects0 = ax1.bar(
        x, cls_list, width=bar_width, label='Vanilla acc.', fc='salmon')
    rects1 = ax1.bar(
        x + bar_width,
        kd_list,
        width=bar_width,
        label='Distill acc.',
        fc='skyblue')
    ax1.set_ylim([68.8, 71.5])
    ax1.set_ylabel(
        'Top-1 Accuracy (%)', fontsize=GLOBAL_FONTSIZE - 2, weight='bold')

    for i, rect in enumerate(rects0):
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            height - 0.18,
            cls_list[i],
            ha='center',
            va='bottom',
            fontsize=GLOBAL_LEGENDSIZE - 2)

    for i, rect in enumerate(rects1):
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            height - 0.18,
            kd_list[i],
            ha='center',
            va='bottom',
            fontsize=GLOBAL_LEGENDSIZE - 2)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(
        x + bar_width * 2,
        pre_list,
        width=bar_width,
        label='DisWOT Score',
        fc='plum')
    ax2.set_ylim([0.0003, 0.0005])
    ax2.set_ylabel('DisWOT Score', fontsize=GLOBAL_FONTSIZE - 2, weight='bold')

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax2.yaxis.set_major_formatter(formatter)

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            height - 0.000014,
            pre_label[i],
            ha='center',
            va='bottom',
            fontsize=GLOBAL_LEGENDSIZE - 2)

    label_size = fig.legend(
        loc='upper right', bbox_to_anchor=(0.83, 0.89)).get_texts()
    [label.set_fontsize(GLOBAL_LEGENDSIZE - 5) for label in label_size]

    ax1.set_xticks(x + bar_width, labels, fontsize=GLOBAL_LEGENDSIZE)
    ax1.tick_params(labelsize=GLOBAL_LEGENDSIZE - 3)
    ax2.tick_params(labelsize=GLOBAL_LEGENDSIZE - 3)
    plt.tight_layout()
    plt.grid(linestyle='-.', alpha=0.9, lw=2)
    plt.savefig(
        './tmp/param_cls_kd_diswot.pdf',
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_time_cost_vs_accuracy():
    # C100
    # RS, RL, BOHB, DARTS, GDAS, NWOT, TE-NAS, DisWOT, DisWOT+
    labels = [
        'RS', 'RL', 'BOHB', 'DARTS', 'GDAS', 'NWOT', 'TE-NAS', 'DisWOT',
        r'DisWOT($M_r$)'
    ]
    time_cost = [216000, 216000, 216000, 23000, 22000, 2200, 2200, 1200, 720]
    acc = [71.28, 71.71, 70.84, 66.24, 70.70, 73.31, 71.24, 74.21, 73.62]
    markers = ['X', ',', 'o', 'v', 'D', 'p', '>', '^', '*']
    x_offset = [-110000, -100000, -110000, 1000, 1000, 100, 100, 100, -160]
    y_offset = [-0.1, 0.2, -0.16, 0.2, 0.2, 0.2, 0.2, -0.2, -0.8]

    plt.figure(figsize=FIGSIZE)
    plt.rc('font', family='Times New Roman')
    plt.grid(linestyle='-.', alpha=0.9, lw=2)

    _len = len(labels)

    ax = plt.gca()
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.annotate(
        '',
        xy=(1200, 73.62),
        xytext=(216000, 71.28),
        arrowprops=dict(arrowstyle='->',
                        lw=2),  # , facecolor='b', shrink=0.01, ec='b'),
        fontsize=GLOBAL_LEGENDSIZE,
        c='b')

    plt.text(22000, 72.5, '180x faster', fontsize=GLOBAL_LEGENDSIZE)

    for i in range(_len):
        plt.scatter(
            time_cost[i], acc[i], label=labels[i], s=100, marker=markers[i])
        plt.text(
            time_cost[i] + x_offset[i],
            acc[i] + y_offset[i],
            s=labels[i],
            fontsize=GLOBAL_LEGENDSIZE - 5)

    plt.xscale('symlog')
    plt.ylim([64, 76])
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE)
    plt.xlabel('Log Time Cost (s)', fontsize=GLOBAL_LEGENDSIZE, weight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=GLOBAL_FONTSIZE, weight='bold')
    plt.tight_layout()
    plt.legend(loc='lower left', fontsize=GLOBAL_LEGENDSIZE - 8)
    plt.savefig(
        './tmp/plot_time_cost_vs_accuracy.pdf',
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_rank_zc_vs_kd_auto(xlabel, ylabel, foldername, file_name):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax1 = axs[0]
    real_rank = [
        0.8033898305084753, 0.9694915254237277, 0.7559322033898311,
        0.6881355932203387, 0.2576271186440693, 1.0, 0.15254237288135675,
        0.8169491525423758, 0.7423728813559307, 0.5152542372881386,
        0.03728813559322011, 0.5796610169491547, 0.6576271186440664,
        0.86440677966102, 0.9322033898305075, 0.7898305084745749,
        0.7796610169491557, 0.5762711864406784, 0.12542372881356073,
        0.9016949152542401, 0.7288135593220352, 0.5525423728813587,
        0.07457627118644022, 0.610169491525427, 0.827118644067795,
        0.6135593220338985, 0.8033898305084753, 0.23389830508474477,
        0.799999999999999, 0.8338983050847476, 0.2474576271186452,
        0.4440677966101698, 0.20000000000000095, 0.5220338983050864,
        0.5796610169491547, 0.691525423728815, 0.5593220338983065,
        0.4237288135593216, 0.6406779661016945, 0.2983050847457657,
        0.8474576271186433, 0.0, 0.4881355932203377, 0.7118644067796632,
        0.5694915254237306, 0.6305084745762704, 0.5152542372881386,
        0.664406779661019, 0.9694915254237277, 0.3118644067796613
    ]
    angle_rank = [
        0.6629917959472171, 0.9450853099942775, 0.6107126584241236,
        0.7620759717004627, 0.41074191682201494, 1.0, 0.010497206375368353,
        0.31694086168882846, 0.7747979991881995, 0.2164592298391797,
        0.23327550201368416, 0.16854236126482366, 0.34668194500654564,
        0.742981863702399, 0.8566542225083371, 0.7525571637502937,
        0.8537823704942521, 0.8667358494624474, 0.07703063049777419,
        0.11320260554464423, 0.7045805350687686, 0.6646442072897527,
        0.20719980384172632, 0.18312970162868714, 0.9001470602091132,
        0.19839948611522826, 0.7444401336987128, 0.0, 0.23016492601093858,
        0.23251906920744964, 0.10672175557812666, 0.20952548988546219,
        0.2790190886180011, 0.6191907780486163, 0.7436998268138469,
        0.2513080534815142, 0.25801666600633727, 0.44214552936038287,
        0.6286101966362793, 0.3032685764886513, 0.4330920483966295,
        0.1665475156323471, 0.27739945579872055, 0.3004227571701373,
        0.5232675583116007, 0.3853188353872021, 0.3316669042305191,
        0.7903212633520121, 0.8668020385255787, 0.4210608971003225
    ]

    ax1.scatter(real_rank, angle_rank, alpha=0.6, marker='*', s=100)
    ax1.set_title('DSS')
    ax1.text(0, 0.9, 'Spearman=0.589', fontsize=20)
    ax1.set_xticks(np.arange(0, 1.1, 0.2))
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    # ax1.set_xlabel('KD Score', fontsize=20)
    # ax1.set_ylabel('ZC Score', fontsize=20)
    ax1.grid(True, lw=2, ls='-.', alpha=0.3, which='both')

    ax2 = axs[1]

    real_rank = [
        0.8033898305084753, 0.9694915254237277, 0.7559322033898311,
        0.6881355932203387, 0.2576271186440693, 1.0, 0.15254237288135675,
        0.8169491525423758, 0.7423728813559307, 0.5152542372881386,
        0.03728813559322011, 0.5796610169491547, 0.6576271186440664,
        0.86440677966102, 0.9322033898305075, 0.7898305084745749,
        0.7796610169491557, 0.5762711864406784, 0.12542372881356073,
        0.9016949152542401, 0.7288135593220352, 0.5525423728813587,
        0.07457627118644022, 0.610169491525427, 0.827118644067795,
        0.6135593220338985, 0.8033898305084753, 0.23389830508474477,
        0.799999999999999, 0.8338983050847476, 0.2474576271186452,
        0.4440677966101698, 0.20000000000000095, 0.5220338983050864,
        0.5796610169491547, 0.691525423728815, 0.5593220338983065,
        0.4237288135593216, 0.6406779661016945, 0.2983050847457657,
        0.8474576271186433, 0.0, 0.4881355932203377, 0.7118644067796632,
        0.5694915254237306, 0.6305084745762704, 0.5152542372881386,
        0.664406779661019, 0.9694915254237277, 0.3118644067796613
    ]
    angle_rank = [
        0.6405390577634446, 0.8738249318307946, 0.6532308034639039,
        0.7241379023512433, 0.39666956268006853, 1.0, 0.08205387850106222,
        0.4126313128749975, 0.6156175311962604, 0.2631681851242462,
        0.11764492884817325, 0.3009176078812906, 0.19464694436423507,
        0.7946174283183303, 0.8486951026141127, 0.5432538644280013,
        0.7789438451835227, 0.8122315923727078, 0.1485865395945208,
        0.38111233080826845, 0.5530495522263237, 0.5859171614744589,
        0.0939262258231963, 0.3348218216475398, 0.7376319993782562,
        0.4065207560897349, 0.6952419172790398, 0.0, 0.3846236845863289,
        0.30568293662619067, 0.10227649820941159, 0.1658522865569955,
        0.29862127758369605, 0.5697178924479323, 0.592831127325028,
        0.41661958359967755, 0.2574701875305086, 0.3599173102972745,
        0.5672245660092778, 0.23201623840536095, 0.4934998872738027,
        0.08833884921843463, 0.30398265752076575, 0.43139276405231225,
        0.48861012532027065, 0.6091505267200107, 0.5086101988485212,
        0.5483324994997308, 0.8564378940680211, 0.34815951710919
    ]

    ax2.scatter(real_rank, angle_rank, alpha=0.6, marker='*', s=100)
    ax2.set_title('NWOT')
    ax2.text(0, 0.9, 'Spearman=0.742', fontsize=20)
    ax2.set_xticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.grid(True, lw=2, ls='-.', alpha=0.3, which='both')

    ax3 = axs[2]
    real_rank = [
        0.8033898305084753, 0.9694915254237277, 0.7559322033898311,
        0.6881355932203387, 0.2576271186440693, 1.0, 0.15254237288135675,
        0.8169491525423758, 0.7423728813559307, 0.5152542372881386,
        0.03728813559322011, 0.5796610169491547, 0.6576271186440664,
        0.86440677966102, 0.9322033898305075, 0.7898305084745749,
        0.7796610169491557, 0.5762711864406784, 0.12542372881356073,
        0.9016949152542401, 0.7288135593220352, 0.5525423728813587,
        0.07457627118644022, 0.610169491525427, 0.827118644067795,
        0.6135593220338985, 0.8033898305084753, 0.23389830508474477,
        0.799999999999999, 0.8338983050847476, 0.2474576271186452,
        0.4440677966101698, 0.20000000000000095, 0.5220338983050864,
        0.5796610169491547, 0.691525423728815, 0.5593220338983065,
        0.4237288135593216, 0.6406779661016945, 0.2983050847457657,
        0.8474576271186433, 0.0, 0.4881355932203377, 0.7118644067796632,
        0.5694915254237306, 0.6305084745762704, 0.5152542372881386,
        0.664406779661019, 0.9694915254237277, 0.3118644067796613
    ]
    angle_rank = [
        0.6277085265424629, 1.0, 0.565275539724974, 0.7731857067503229,
        0.2956360814094417, 0.9430730757415995, 0.0, 0.20036174533597567,
        0.7333329368161068, 0.19114830309757624, 0.1402084170845732,
        0.06456086800235114, 0.3236734888424797, 0.7182245798321805,
        0.8957243961729044, 0.6154750306436827, 0.8653595587354836,
        0.7050847569576457, 0.16841671926921423, 0.2319810633894987,
        0.5716006295730591, 0.6335318637614948, 0.14004655390881757,
        0.2002898029333405, 0.7700220689806441, 0.1803435734187048,
        0.692255122380937, 0.17298605009227327, 0.17587500030896605,
        0.21165927015256764, 0.004292832359286318, 0.19867130508255432,
        0.25668541919920235, 0.5898293118253409, 0.684144221738474,
        0.16934016571049165, 0.22438868660477496, 0.41917740133821185,
        0.6226360529162205, 0.3056078467985384, 0.2801376789633625,
        0.06799273489849242, 0.24791553006280032, 0.19178644254512558,
        0.34746573433018924, 0.37409433834874817, 0.57942698942207,
        0.6586069919666394, 0.9044980873899929, 0.3313857870632459
    ]

    ax3.scatter(real_rank, angle_rank, alpha=0.6, marker='*', s=100)
    ax3.set_title('Synflow')
    ax3.text(0, 0.9, 'Spearman=0.598', fontsize=20)
    ax3.set_xticks(np.arange(0, 1.1, 0.2))
    ax3.set_yticks(np.arange(0, 1.1, 0.2))
    ax3.grid(True, lw=2, ls='-.', alpha=0.3, which='both')

    ax4 = axs[3]
    real_rank = [
        0.8033898305084753, 0.9694915254237277, 0.7559322033898311,
        0.6881355932203387, 0.2576271186440693, 1.0, 0.15254237288135675,
        0.8169491525423758, 0.7423728813559307, 0.5152542372881386,
        0.03728813559322011, 0.5796610169491547, 0.6576271186440664,
        0.86440677966102, 0.9322033898305075, 0.7898305084745749,
        0.7796610169491557, 0.5762711864406784, 0.12542372881356073,
        0.9016949152542401, 0.7288135593220352, 0.5525423728813587,
        0.07457627118644022, 0.610169491525427, 0.827118644067795,
        0.6135593220338985, 0.8033898305084753, 0.23389830508474477,
        0.799999999999999, 0.8338983050847476, 0.2474576271186452,
        0.4440677966101698, 0.20000000000000095, 0.5220338983050864,
        0.5796610169491547, 0.691525423728815, 0.5593220338983065,
        0.4237288135593216, 0.6406779661016945, 0.2983050847457657,
        0.8474576271186433, 0.0, 0.4881355932203377, 0.7118644067796632,
        0.5694915254237306, 0.6305084745762704, 0.5152542372881386,
        0.664406779661019, 0.9694915254237277, 0.3118644067796613
    ]
    angle_rank = [
        0.6671910329583628, 1.0, 0.5170676664860131, 0.7793227379479976,
        0.17496808484895734, 0.9973490075948243, 0.0032183199082134,
        0.4948685104848312, 0.25583530579721003, 0.3353680897729058,
        0.04680364171342806, 0.2848269953451579, 0.1971016945663489,
        0.852472086690966, 0.8307759611978222, 0.47681796246202285,
        0.6227303981555786, 0.7317145303816411, 0.0, 0.4457117481916326,
        0.4284263235078839, 0.4446141954711881, 0.017593019477136353,
        0.20271718402222022, 0.6930779311201521, 0.5176800216991119,
        0.8108798117899488, 0.10682517340466596, 0.4374994767111204,
        0.4468502322280652, 0.010669807139968714, 0.2404552788709784,
        0.1989490274497632, 0.309717364484771, 0.4104593918957108,
        0.5799370800872812, 0.33290664710200846, 0.21468158139240204,
        0.4742605465616178, 0.18994934118039472, 0.581141063459709,
        0.03611995553200314, 0.15127126473533142, 0.6503156789650235,
        0.4353226315754336, 0.4347655076584864, 0.5025699247028182,
        0.36694558113146775, 0.951132838762977, 0.17570579385845334
    ]

    ax4.scatter(real_rank, angle_rank, alpha=0.6, marker='*', s=100)
    ax4.set_title('TVT')  # 0.851
    ax4.text(0, 0.9, 'Spearman=0.848', fontsize=20)
    ax4.set_xticks(np.arange(0, 1.1, 0.2))
    ax4.set_yticks(np.arange(0, 1.1, 0.2))
    ax4.grid(True, lw=2, ls='-.', alpha=0.3, which='both')

    fig.tight_layout()

    save_path = os.path.join(foldername, file_name)
    # foldername / '{}'.format(file_name)
    print('save figure into {:}\n'.format(save_path))
    plt.savefig(
        str(save_path),
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_hist_rank_consistency_tvt():
    """Plot the histogram of the rank consistency of TVT."""
    result = {
        'CIFAR-100': {
            'NWOT': 0.39,
            'DSS': 0.48,
            'TVT': 0.67
        },
        'Flowers': {
            'NWOT': 0.62,
            'DSS': 0.63,
            'TVT': 0.72
        },
        'Chaoyang': {
            'NWOT': 0.19,
            'DSS': 0.19,
            'TVT': 0.24,
        }
    }

    labels = result.keys()
    x = np.arange(len(labels))
    y1 = [v['NWOT'] for k, v in result.items()]
    y2 = [v['DSS'] for k, v in result.items()]
    y3 = [v['TVT'] for k, v in result.items()]
    width = 0.3

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    rects1 = ax.bar(
        x - width / 2, y1, width, label='NWOT', color='navajowhite')
    rects2 = ax.bar(x + width / 2, y2, width, label='DSS', color='palegreen')
    rects3 = ax.bar(x + width * 3 / 2, y3, width, label='TVT', color='plum')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Kendall's Tau", fontsize=GLOBAL_FONTSIZE - 6, weight='bold')
    ax.set_xticks(x, labels)  #, fontsize=GLOBAL_LEGENDSIZE - 2)
    yticks = np.arange(0, 0.81, 0.1)
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE - 2)
    ax.set_yticks(yticks)  #, fontsize=GLOBAL_LABELSIZE)
    ax.set_ylim(0, 0.85)

    ax.legend(
        loc='upper right',
        fancybox=True,
        shadow=False,
        ncol=1,
        fontsize=GLOBAL_LEGENDSIZE - 4)

    ax.bar_label(
        rects1, padding=-26, label_type='edge', fontsize=GLOBAL_LEGENDSIZE - 4)
    ax.bar_label(
        rects2, padding=-26, label_type='edge', fontsize=GLOBAL_LEGENDSIZE - 4)
    ax.bar_label(
        rects3, padding=-26, label_type='edge', fontsize=GLOBAL_LEGENDSIZE - 4)

    fig.tight_layout()
    plt.grid(linestyle='-.', alpha=0.4, lw=2)

    plt.savefig(
        './hist_rank_consistency.pdf',
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


def plot_rank_zc_vs_kd_pit(xlabel, ylabel, foldername, file_name):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax1 = axs[0]
    real_rank = [
        0.5719584569732937, 0.9681008902077156, 0.0, 0.26186943620177966,
        0.42136498516320453, 0.15207715133531147, 0.761127596439169,
        0.9272997032640956, 0.771513353115727, 0.9235905044510384,
        0.3924332344213647, 0.7982195845697329, 0.7410979228486648,
        0.7010385756676555, 0.6676557863501489, 0.7678041543026708,
        0.8360534124629083, 0.7908011869436206, 0.9124629080118699,
        0.8523738872403563, 0.8553412462908009, 0.9213649851632045, 1.0,
        0.7908011869436206, 0.7470326409495549, 0.8842729970326407,
        0.7396142433234426, 0.7366468842729971, 0.26409495548961354,
        0.6839762611275969, 0.9258160237388724, 0.9050445103857565,
        0.7529673590504451, 0.8137982195845702, 0.7270029673590508,
        0.8034124629080124, 0.8620178041543026, 0.6721068249258156,
        0.8382789317507423, 0.9799703264094959, 0.9399109792284864,
        0.8442136498516324, 0.9005934718100898, 0.9473293768545998,
        0.5385756676557861, 0.14391691394658748, 0.844955489614244,
        0.7433234421364988, 0.9873887240356082, 0.7626112759643913
    ]
    angle_rank = [
        0.11799876028008148, 1.0, 0.0, 0.18260817166922136,
        0.05006264320223978, 0.022503100839024147, 0.19562924571688012,
        0.6863309468374514, 0.2841796410751825, 0.9187902596527244,
        0.04578109493732689, 0.30835001886322755, 0.3986677420294492,
        0.2812549793341532, 0.2726100468983199, 0.20901025078091848,
        0.47901377668436607, 0.20048193624667945, 0.7022974806666191,
        0.3622961385320239, 0.3039929319273466, 0.7910494632112127,
        0.7254645724830043, 0.3532780658710778, 0.2501747337213804,
        0.5816126886772063, 0.2333158486908103, 0.3100855676775887,
        0.20532768326687229, 0.433658785797338, 0.986595904086448,
        0.5411329949133231, 0.3382240676033904, 0.5029165558472988,
        0.32080586056076504, 0.24254784537664914, 0.5280274366837372,
        0.2882058305647781, 0.3100962253346457, 0.7878617887806144,
        0.5772971333747311, 0.555805521336991, 0.6782854147494486,
        0.7060075638997502, 0.2791444177175606, 0.1333067608026116,
        0.5083032808198982, 0.26178098315881226, 0.9237399656568785,
        0.4830181344328153
    ]

    ax1.scatter(real_rank, angle_rank, alpha=0.6, marker='^', s=100)
    ax1.set_title('DSS')
    ax1.text(0, 0.9, 'Spearman=0.877', fontsize=20)
    ax1.set_xticks(np.arange(0, 1.1, 0.2))
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    # ax1.set_xlabel('KD Score', fontsize=20)
    # ax1.set_ylabel('ZC Score', fontsize=20)
    ax1.grid(True, lw=2, ls='-.', alpha=0.3, which='both')

    ax2 = axs[1]

    real_rank = [
        0.5719584569732937, 0.9681008902077156, 0.0, 0.26186943620177966,
        0.42136498516320453, 0.15207715133531147, 0.761127596439169,
        0.9272997032640956, 0.771513353115727, 0.9235905044510384,
        0.3924332344213647, 0.7982195845697329, 0.7410979228486648,
        0.7010385756676555, 0.6676557863501489, 0.7678041543026708,
        0.8360534124629083, 0.7908011869436206, 0.9124629080118699,
        0.8523738872403563, 0.8553412462908009, 0.9213649851632045, 1.0,
        0.7908011869436206, 0.7470326409495549, 0.8842729970326407,
        0.7396142433234426, 0.7366468842729971, 0.26409495548961354,
        0.6839762611275969, 0.9258160237388724, 0.9050445103857565,
        0.7529673590504451, 0.8137982195845702, 0.7270029673590508,
        0.8034124629080124, 0.8620178041543026, 0.6721068249258156,
        0.8382789317507423, 0.9799703264094959, 0.9399109792284864,
        0.8442136498516324, 0.9005934718100898, 0.9473293768545998,
        0.5385756676557861, 0.14391691394658748, 0.844955489614244,
        0.7433234421364988, 0.9873887240356082, 0.7626112759643913
    ]
    angle_rank = [
        0.42050450528641026, 0.6479917777723954, 0.018720853532247915,
        0.13484712648974442, 0.506197174694155, 0.34803205055021647,
        0.28601755471088913, 0.6044477519850442, 0.5269270777171147,
        0.6211189782181425, 0.5034812869190785, 0.449938989508439,
        0.1341065917092119, 0.3828670767486377, 0.24765017359429894,
        0.43645154682615067, 0.5540277970404639, 0.6162342484052944,
        0.6792944602055059, 0.857333992667885, 0.843332446067859,
        0.6636095962914578, 0.8068896743089585, 0.4110587851937643,
        0.4042490036265982, 0.3467760326153979, 0.7340917829467635,
        0.4065830101704958, 0.0, 0.17439697988981862, 0.6412405263945139,
        0.47435849714541595, 0.6767296211382374, 0.046850366216022184,
        0.37876604698739513, 0.7314275737896522, 0.13760474453506458,
        0.3440493730400513, 0.8981072604865163, 0.6631287260263564,
        0.9576356173882792, 0.6683625532496136, 0.5349446950215507, 1.0,
        0.07260275169625034, 0.08069965837295817, 0.7266772674936514,
        0.611848248297926, 0.4477885589774272, 0.39757468836833293
    ]

    ax2.scatter(real_rank, angle_rank, alpha=0.6, marker='^', s=100)
    ax2.set_title('NWOT')
    ax2.text(0, 0.9, 'Spearman=0.644', fontsize=20)
    ax2.set_xticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.grid(True, lw=2, ls='-.', alpha=0.3, which='both')

    ax3 = axs[2]
    real_rank = [
        0.5719584569732937, 0.9681008902077156, 0.0, 0.26186943620177966,
        0.42136498516320453, 0.15207715133531147, 0.761127596439169,
        0.9272997032640956, 0.771513353115727, 0.9235905044510384,
        0.3924332344213647, 0.7982195845697329, 0.7410979228486648,
        0.7010385756676555, 0.6676557863501489, 0.7678041543026708,
        0.8360534124629083, 0.7908011869436206, 0.9124629080118699,
        0.8523738872403563, 0.8553412462908009, 0.9213649851632045, 1.0,
        0.7908011869436206, 0.7470326409495549, 0.8842729970326407,
        0.7396142433234426, 0.7366468842729971, 0.26409495548961354,
        0.6839762611275969, 0.9258160237388724, 0.9050445103857565,
        0.7529673590504451, 0.8137982195845702, 0.7270029673590508,
        0.8034124629080124, 0.8620178041543026, 0.6721068249258156,
        0.8382789317507423, 0.9799703264094959, 0.9399109792284864,
        0.8442136498516324, 0.9005934718100898, 0.9473293768545998,
        0.5385756676557861, 0.14391691394658748, 0.844955489614244,
        0.7433234421364988, 0.9873887240356082, 0.7626112759643913
    ]
    angle_rank = [
        0.11361064413929278, 1.0, 0.0, 0.19696319233172682,
        0.04621769495920202, 0.012264654017141921, 0.19476598974520076,
        0.6833302199939996, 0.27231651147364383, 0.9180103071787765,
        0.04270348997723709, 0.30039883229983516, 0.36614592024725023,
        0.2651672017122576, 0.27165513046677525, 0.18173819856235104,
        0.4613530271325223, 0.18687207661352973, 0.6778838571576528,
        0.3866714116231186, 0.29613729111080644, 0.7524071303029088,
        0.7225895131398066, 0.3539731440188732, 0.24653479741699855,
        0.5629347346024979, 0.23210772190616702, 0.29987331647255056,
        0.1977752596788278, 0.41548473664219554, 0.99695606146629,
        0.52126615541658, 0.326209039873575, 0.4534233616564774,
        0.31434705986392497, 0.23373847465695902, 0.5268786977190686,
        0.25182055879326914, 0.3097414649831343, 0.7847876010710106,
        0.5723992793043738, 0.5311191184737426, 0.6752222203860632,
        0.6924347355374803, 0.2632186546892593, 0.11230080826938665,
        0.5145614436180491, 0.24026048887604648, 0.9436041259138787,
        0.4544164232205254
    ]

    ax3.scatter(real_rank, angle_rank, alpha=0.6, marker='^', s=100)
    ax3.set_title('Synflow')
    ax3.text(0, 0.9, 'Spearman=0.875', fontsize=20)
    ax3.set_xticks(np.arange(0, 1.1, 0.2))
    ax3.set_yticks(np.arange(0, 1.1, 0.2))
    ax3.grid(True, lw=2, ls='-.', alpha=0.3, which='both')

    ax4 = axs[3]
    real_rank = [
        0.5719584569732937, 0.9681008902077156, 0.0, 0.26186943620177966,
        0.42136498516320453, 0.15207715133531147, 0.761127596439169,
        0.9272997032640956, 0.771513353115727, 0.9235905044510384,
        0.3924332344213647, 0.7982195845697329, 0.7410979228486648,
        0.7010385756676555, 0.6676557863501489, 0.7678041543026708,
        0.8360534124629083, 0.7908011869436206, 0.9124629080118699,
        0.8523738872403563, 0.8553412462908009, 0.9213649851632045, 1.0,
        0.7908011869436206, 0.7470326409495549, 0.8842729970326407,
        0.7396142433234426, 0.7366468842729971, 0.26409495548961354,
        0.6839762611275969, 0.9258160237388724, 0.9050445103857565,
        0.7529673590504451, 0.8137982195845702, 0.7270029673590508,
        0.8034124629080124, 0.8620178041543026, 0.6721068249258156,
        0.8382789317507423, 0.9799703264094959, 0.9399109792284864,
        0.8442136498516324, 0.9005934718100898, 0.9473293768545998,
        0.5385756676557861, 0.14391691394658748, 0.844955489614244,
        0.7433234421364988, 0.9873887240356082, 0.7626112759643913
    ]
    angle_rank = [
        0.30391403572101333, 0.94506424920679, 0.1002390596902867,
        0.1201292320638773, 0.17906570591723936, 0.08983764018494719,
        0.301297118646088, 0.6181323319365201, 0.40962109624768833,
        0.8039363347255951, 0.148411162357342, 0.35916939805337816,
        0.3226818825142494, 0.3467071229547988, 0.2145865224708231,
        0.37607124737036857, 0.49743209960504053, 0.3719413372756035,
        0.7666103769388003, 0.7940946414638598, 0.591093807100062,
        0.7633427417397137, 0.9263125867507016, 0.45249977125837365,
        0.35733030614496025, 0.5231155592215362, 0.43912509770861774,
        0.4032382678201601, 0.0, 0.28190544793513594, 0.9274152133168078,
        0.564277810693586, 0.4836551858542724, 0.268078535890499,
        0.3466676979072159, 0.6115363760979455, 0.36629463277606833,
        0.27720443236063796, 0.7021646768519685, 0.9656373852176314,
        0.8943899543187289, 0.5029821535872032, 0.6872880502046067, 1.0,
        0.1781778457609514, 0.03774376193618435, 0.8162929907841928,
        0.4888609308475625, 0.6299228789459043, 0.2153156695552062
    ]

    ax4.scatter(real_rank, angle_rank, alpha=0.6, marker='^', s=100)
    ax4.set_title('TVT')  # 0.851
    ax4.text(0, 0.9, 'Spearman=0.909', fontsize=20)
    ax4.set_xticks(np.arange(0, 1.1, 0.2))
    ax4.set_yticks(np.arange(0, 1.1, 0.2))
    ax4.grid(True, lw=2, ls='-.', alpha=0.3, which='both')

    fig.tight_layout()

    save_path = os.path.join(foldername, file_name)
    # foldername / '{}'.format(file_name)
    print('save figure into {:}\n'.format(save_path))
    plt.savefig(
        str(save_path),
        dpi=GLOBAL_DPI,
        bbox_inches='tight',
        pad_inches=PADINCHES,
        format='pdf')
    plt.clf()


if __name__ == '__main__':
    # plot_standalone_model_rank(xlabel='Ranking at ground-truth setting',
    #                            ylabel='Ranking by DisWOT',
    #                            foldername='./',
    #                            file_name='standalone_ranks.pdf')
    # plot_ranking_stability(xlabel='Datasets',
    #                        ylabel='Ranking Correlation',
    #                        foldername='./tmp',
    #                        file_name='ranking_stability.pdf')

    # plot_evolution_search_process()  # fig6
    # plot_kd_box(foldername='./tmp', file_name='kd_box.pdf')  # fig5
    # plot_different_teacher_diswot()  # fig3
    # plot_hist_rank_consistency()  # fig1
    # plot_param_cls_kd_ours()  # fig2
    # plot_time_cost_vs_accuracy()  # fig4

    # plot_kd_zc_box(xlabel='Datasets',
    #                ylabel='Ranking Correlation',
    #                foldername='./tmp',
    #                file_name='kd_zc_box.pdf')

    # fig in tvt autoformer
    # plot_rank_zc_vs_kd_auto(
    #     xlabel='Ranking at ground-truth setting',
    #     ylabel='Ranking by TVT',
    #     foldername='./',
    #     file_name='standalone_ranks.pdf')

    # fig in tvt pit
    plot_rank_zc_vs_kd_pit(
        xlabel='Ranking at ground-truth setting',
        ylabel='Ranking by TVT',
        foldername='./',
        file_name='rank_pit.pdf')

    # plot_hist_rank_consistency_tvt()

    dss = [
        736.8451538, 2403.74707, 513.8382568, 858.9510498, 608.4520874,
        556.3670654, 883.5596924, 1810.941162, 1050.911865, 2250.268066,
        600.3603516, 1096.591675, 1267.283936, 1045.384521, 1029.046387,
        908.8485718, 1419.130615, 892.730835, 1841.116455, 1198.544922,
        1088.357178, 2008.849609, 1884.900146, 1181.501587, 986.6456909,
        1613.033203, 954.7839355, 1099.871704, 901.888855, 1333.413818,
        2378.414551, 1536.530273, 1153.050903, 1464.304688, 1120.13208,
        972.2315674, 1511.761963, 1058.520996, 1099.891846, 2002.825195,
        1604.877197, 1564.26001, 1795.73584, 1848.128174, 1041.395752,
        765.7758789, 1474.485107, 1008.580444, 2259.622559, 1426.698486
    ]
    nwot = [
        2543.131682, 2665.515008, 2326.980602, 2389.454073, 2589.23252,
        2504.143039, 2470.780556, 2642.089246, 2600.384768, 2651.058012,
        2587.77143, 2558.96681, 2389.055681, 2522.883544, 2450.139719,
        2551.710852, 2614.96438, 2648.430131, 2682.355187, 2778.136678,
        2770.604143, 2673.917063, 2750.998705, 2538.050085, 2534.386567,
        2503.467328, 2711.834984, 2535.642213, 2316.90918, 2410.731055,
        2661.882978, 2572.103987, 2680.975358, 2342.113683, 2520.677277,
        2710.401696, 2390.937613, 2502.000443, 2800.071831, 2673.658365,
        2832.096824, 2676.474053, 2604.698076, 2854.887963, 2355.96792,
        2360.323884, 2707.846132, 2646.070556, 2557.809924, 2530.795927
    ]
    synflow = [
        744.3295692, 2403.757332, 531.6367432, 900.3755908, 618.1618415,
        554.5976545, 896.2621626, 1810.913317, 1041.446091, 2250.26274,
        611.582826, 1094.019582, 1217.106059, 1028.061721, 1040.207906,
        871.8725665, 1395.345244, 881.4838053, 1800.717069, 1255.532254,
        1086.041463, 1940.233623, 1884.411448, 1194.317154, 993.1796133,
        1585.51845, 966.1703882, 1093.035753, 901.8958788, 1309.474273,
        2398.058712, 1507.509845, 1142.339403, 1380.499954, 1120.132346,
        969.223354, 1518.017201, 1003.075196, 1111.510117, 2000.853769,
        1603.237219, 1525.95578, 1795.734164, 1827.958068, 1024.413806,
        741.8773985, 1494.957816, 981.4333511, 2298.177455, 1382.359085
    ]
    tvt = [
        2.211442947, 8.92689991, 0.078135483, 0.286466688, 0.903771877,
        -0.03080979, 2.184033155, 5.502589703, 3.318626881, 7.44871521,
        0.582693815, 2.790191889, 2.408018827, 2.659661055, 1.275819659,
        2.967223167, 4.238366127, 2.923966169, 7.057760239, 7.345632553,
        5.219386101, 7.023534775, 8.730493546, 3.767741442, 2.770929098,
        4.507376671, 3.627654076, 3.251772642, -0.971776187, 1.980923295,
        8.742042542, 4.938513279, 4.094065666, 1.836099148, 2.659248114,
        5.433503151, 2.864822149, 1.931684494, 6.382750988, 9.142384529,
        8.396133423, 4.296497822, 6.226932049, 9.502301216, 0.894472361,
        -0.576445103, 7.578139782, 4.148591042, 5.626084805, 1.283456802
    ]
    kd = [
        72.5, 77.84, 64.79, 68.32, 70.47, 66.84, 75.05, 77.29, 75.19, 77.24,
        70.08, 75.55, 74.78, 74.24, 73.79, 75.14, 76.06, 75.45, 77.09, 76.28,
        76.32, 77.21, 78.27, 75.45, 74.86, 76.71, 74.76, 74.72, 68.35, 74.01,
        77.27, 76.99, 74.94, 75.76, 74.59, 75.62, 76.41, 73.85, 76.09, 78,
        77.46, 76.17, 76.93, 77.56, 72.05, 66.73, 76.18, 74.81, 78.1, 75.07
    ]

    def min_max_scale(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    dss = min_max_scale(dss)
    synflow = min_max_scale(synflow)
    tvt = min_max_scale(tvt)
    nwot = min_max_scale(nwot)
    kd = min_max_scale(kd)

    print(f'dss: {dss.tolist()}')
    print(f'synflow: {synflow.tolist()}')
    print(f'tvt: {tvt.tolist()}')
    print(f'nwot: {nwot.tolist()}')
    print(f'kd: {kd.tolist()}')
