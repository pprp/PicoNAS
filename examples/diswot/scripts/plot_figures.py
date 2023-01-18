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
    # result = {
    #     'Fisher': {
    #         'cls': 0.8168,
    #         'kd': 0.6286
    #     },
    #     'Nwot': {
    #         'cls': 0.4029,
    #         'kd': 0.3187
    #     },
    #     'Snip': {
    #         'cls': 0.8466,
    #         'kd': 0.6722
    #     },
    #     'Vanilla': {
    #         'cls': 1,
    #         'kd': 0.8521
    #     }
    # }

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
    ax.set_xticks(x, labels, fontsize=GLOBAL_LEGENDSIZE - 2)
    yticks = np.arange(0.1, 1.2, 0.1)
    plt.tick_params(labelsize=GLOBAL_LEGENDSIZE - 2)
    ax.set_yticks(yticks, fontsize=GLOBAL_LABELSIZE)
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
        './tmp/hist_rank_consistency.pdf',
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


def plot_rank_zc_vs_kd(xlabel, ylabel, foldername, file_name):
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

    # fig in tvt
    plot_rank_zc_vs_kd(
        xlabel='Ranking at ground-truth setting',
        ylabel='Ranking by TVT',
        foldername='./',
        file_name='standalone_ranks.pdf')

    dss = [
        4145.683105, 4799.123047, 4024.583984, 4375.201172, 3561.372803,
        4926.327148, 2634.248047, 3344.092529, 4404.67041, 3111.337402,
        3150.290527, 3000.343018, 3412.984619, 4330.97168, 4594.281738,
        4353.151855, 4587.629395, 4617.634766, 2788.365723, 2872.154297,
        4242.019043, 4149.510742, 3089.888916, 3034.133057, 4695.02832,
        3069.503906, 4334.349609, 2609.932373, 3143.085205, 3148.53833,
        2857.14209, 3095.276123, 3256.250732, 4044.222656, 4332.634766,
        3192.061035, 3207.60083, 3634.115967, 4066.041748, 3312.422119,
        3613.144531, 2995.722168, 3252.499023, 3305.830078, 3822.026611,
        3502.48291, 3378.203857, 4440.628418, 4617.788086, 3585.275635
    ]

    gt_kd = [
        76.75, 77.24, 76.61, 76.41, 75.14, 77.33, 74.83, 76.79, 76.57, 75.9,
        74.49, 76.09, 76.32, 76.93, 77.13, 76.71, 76.68, 76.08, 74.75, 77.04,
        76.53, 76.01, 74.6, 76.18, 76.82, 76.19, 76.75, 75.07, 76.74, 76.84,
        75.11, 75.69, 74.97, 75.92, 76.09, 76.42, 76.03, 75.63, 76.27, 75.26,
        76.88, 74.38, 75.82, 76.48, 76.06, 76.24, 75.9, 76.34, 77.24, 75.3
    ]

    nwot = [
        2711.494266, 2734.211071, 2712.730157, 2719.634916, 2687.746854,
        2746.497689, 2657.110352, 2689.30117, 2709.067469, 2674.746818,
        2660.576121, 2678.422764, 2668.074388, 2726.498039, 2731.76399,
        2702.020873, 2724.971784, 2728.213263, 2663.589139, 2686.231929,
        2702.974753, 2706.17532, 2658.266452, 2681.724273, 2720.948938,
        2688.706139, 2716.821096, 2649.120147, 2686.573856, 2678.8868,
        2659.079581, 2665.270435, 2678.199153, 2704.597875, 2706.848585,
        2689.689538, 2674.191961, 2684.16801, 2704.355081, 2671.713318,
        2697.175953, 2657.722367, 2678.721231, 2691.128114, 2696.6998,
        2708.437728, 2698.647358, 2702.515418, 2732.517964, 2683.023065
    ]

    tvt = [
        0.74492836, 1.163174391, 0.556266069, 0.885845959, 0.126344383,
        1.159842849, -0.089496128, 0.528368056, 0.227971435, 0.32792148,
        -0.034721799, 0.264405727, 0.154160023, 0.977773845, 0.950507998,
        0.505683661, 0.689054012, 0.826016009, -0.093540639, 0.466592014,
        0.444869161, 0.465212703, -0.071431227, 0.161217093, 0.777460814,
        0.557035625, 0.925504208, 0.040708162, 0.456271529, 0.468022764,
        -0.080131732, 0.208643124, 0.156481594, 0.295685828, 0.422289848,
        0.635275006, 0.324828148, 0.176252931, 0.502469718, 0.145171553,
        0.63678807, -0.048148148, 0.096564233, 0.723720849, 0.453535855,
        0.452835709, 0.538046539, 0.367605388, 1.101762295, 0.127271473
    ]

    synflow = [
        4198.71084, 5138.428212, 4041.120977, 4565.916224, 3360.512096,
        4994.736445, 2614.283989, 3120.025931, 4465.322085, 3096.769874,
        2968.190255, 2777.244931, 3431.282556, 4427.186413, 4875.221549,
        4167.831732, 4798.57632, 4394.019605, 3039.392078, 3199.83765,
        4057.086416, 4213.409783, 2967.781689, 3119.844338, 4557.930746,
        3069.497178, 4361.635757, 3050.925728, 3058.217855, 3148.542513,
        2625.119717, 3115.759016, 3262.195007, 4103.098239, 4341.162674,
        3041.72299, 3180.673396, 3672.348205, 4185.907185, 3385.68227,
        3321.391893, 2785.907458, 3240.058542, 3098.38063, 3491.337615,
        3558.552052, 4076.841277, 4276.703023, 4897.367611, 3450.749509
    ]

    def min_max_scale(x):
        x = np.array(x)
        return (x - x.min()) / (x.max() - x.min())

    dss = min_max_scale(dss)
    gt_kd = min_max_scale(gt_kd)
    nwot = min_max_scale(nwot)
    tvt = min_max_scale(tvt)
    synflow = min_max_scale(synflow)

    print('dss:', dss.tolist())
    print('nwot:', nwot.tolist())
    print('tvt:', tvt.tolist())
    print('synflow:', synflow.tolist())
    print('kd:', gt_kd.tolist())

    # import numpy as np

    # dss = np.array(dss)
    # dss_idx = np.argsort(dss)

    # gt_kd = np.array(gt_kd)
    # gt_kd_idx = np.argsort(gt_kd)

    # nwot = np.array(nwot)
    # nwot_idx = np.argsort(nwot)

    # tvt = np.array(tvt)
    # tvt_idx = np.argsort(tvt)

    # print("dss:", dss_idx.tolist())

    # print("kd:", gt_kd_idx.tolist())

    # print("nwot:", nwot_idx.tolist())

    # print("tvt:", tvt_idx.tolist())
