# draw line with
import csv
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('agg')


def process_csv(csv_path) -> List:
    epoch_list = []
    train_loss_list = []
    with open(csv_path, 'r') as f:
        cr = csv.reader(f)
        for i, contents in enumerate(cr):
            if i == 0:
                continue
            epoch = int(contents[1])
            train_loss = float(contents[2])
            epoch_list.append(epoch)
            train_loss_list.append(train_loss)
    return epoch_list, train_loss_list


def plot_train_curve():
    train_max_subnet_csv_path = './data/csv/run-graduate_nb201_spos_maxsubnet_exp1.0-tag-STEP_LOSS_train_step_loss.csv'
    x, y = process_csv(train_max_subnet_csv_path)

    train_min_subnet_csv_path = './data/csv/run-graduate_nb201_spos_minsubnet_exp1.2-tag-STEP_LOSS_train_step_loss.csv'
    x3, y3 = process_csv(train_min_subnet_csv_path)

    # figsize setting
    dpi = 100
    width, height = 1600, 1000
    legend_fontsize = 12
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)

    # plot range setting
    plt.xlim(0, max(x))
    plt.ylim(0, 0.7)

    # plot grid background
    interval_x = 2000  # TODO
    interval_y = 0.05  # TODO
    plt.xticks(np.arange(0, max(x) + interval_x, interval_x))
    plt.yticks(np.arange(0, 0.7 + interval_y, interval_y))
    plt.grid()

    # plot title and label
    # plt.title("EXP", fontsize=20)  # TODO
    plt.xlabel('The searching iter', fontsize=16)  # TODO
    plt.ylabel('The training loss', fontsize=16)  # TODO

    # plot curve [linestyle: '-' or ':'] [color: 'g' or 'y']
    plt.plot(
        x, y, color='g', linestyle='-', label='Train loss (Max subnet)', lw=2
    )  # TODO
    plt.legend(loc=2, fontsize=legend_fontsize)  # TODO

    plt.plot(
        x3, y3, color='r', linestyle='-', label='Train loss (Min subnet)', lw=2
    )  # TODO
    plt.legend(loc=2, fontsize=legend_fontsize)  # TODO

    # save figure
    fig.savefig(
        './train_loss_chapt4_exp1.0.0.png', dpi=dpi, bbox_inches='tight'
    )  # TODO
    plt.close(fig)


def plot_valid_curve():
    valid_max_subnet_csv_path = './data/csv/run-graduate_nb201_spos_maxsubnet_exp1.0-tag-STEP_LOSS_valid_step_loss.csv'
    x2, y2 = process_csv(valid_max_subnet_csv_path)

    valid_min_subnet_csv_path = './data/csv/run-graduate_nb201_spos_minsubnet_exp1.2-tag-STEP_LOSS_valid_step_loss.csv'
    x4, y4 = process_csv(valid_min_subnet_csv_path)

    # figsize setting
    dpi = 100
    width, height = 1600, 1000
    legend_fontsize = 12
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)

    # plot range setting
    plt.xlim((0, max(x2)))
    plt.ylim((0.4, 0.8))

    # plot grid background
    interval_x = 1000  # TODO
    interval_y = 0.05  # TODO
    plt.xticks(np.arange(0, max(x2) + interval_x, interval_x))
    plt.yticks(np.arange(0.25, 0.8 + interval_y, interval_y))
    plt.grid()

    # plot title and label
    # plt.title("EXP", fontsize=20)  # TODO
    plt.xlabel('The searching iter', fontsize=16)  # TODO
    plt.ylabel('The validation loss', fontsize=16)  # TODO

    # plot curve [linestyle: '-' or ':'] [color: 'g' or 'y']

    plt.plot(
        x2, y2, color='g', linestyle=':', label='Valid loss (Max subnet)', lw=2
    )  # TODO
    plt.legend(loc=2, fontsize=legend_fontsize)  # TODO

    plt.plot(
        x4, y4, color='r', linestyle=':', label='Valid loss (Min subnet)', lw=2
    )  # TODO
    plt.legend(loc=2, fontsize=legend_fontsize)  # TODO

    # save figure
    fig.savefig('./val_loss_chapt4_exp1.0.1.png', dpi=dpi, bbox_inches='tight')  # TODO
    plt.close(fig)


def plot_valid_loss_alternet_curve():
    valid_max_subnet_csv_path = './data/csv/run-graduate_nb201_spos_altersubnet_exp1.3-tag-STEP_LOSS_valid_step_loss_type_max.csv'
    x2, y2 = process_csv(valid_max_subnet_csv_path)

    valid_min_subnet_csv_path = './data/csv/run-graduate_nb201_spos_altersubnet_exp1.3-tag-STEP_LOSS_valid_step_loss_type_min.csv'
    x4, y4 = process_csv(valid_min_subnet_csv_path)

    # figsize setting
    dpi = 100
    width, height = 1600, 1000
    legend_fontsize = 12
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)

    # plot range setting
    plt.xlim((0, max(x2)))
    plt.ylim((0.4, 0.8))

    # plot grid background
    interval_x = 1000  # TODO
    interval_y = 0.05  # TODO
    plt.xticks(np.arange(0, max(x2) + interval_x, interval_x))
    plt.yticks(np.arange(0.25, 0.8 + interval_y, interval_y))
    plt.grid()

    # plot title and label
    # plt.title("EXP", fontsize=20)  # TODO
    plt.xlabel('The searching iter', fontsize=16)  # TODO
    plt.ylabel('The validation loss', fontsize=16)  # TODO

    # plot curve [linestyle: '-' or ':'] [color: 'g' or 'y']

    plt.plot(
        x2, y2, color='g', linestyle='-', label='Valid loss (Max subnet)', lw=2
    )  # TODO
    plt.legend(loc=2, fontsize=legend_fontsize)  # TODO

    plt.plot(
        x4, y4, color='r', linestyle='-', label='Valid loss (Min subnet)', lw=2
    )  # TODO
    plt.legend(loc=2, fontsize=legend_fontsize)  # TODO

    # save figure
    fig.savefig(
        './val_loss_alternet_chapt4_exp1.0.2.png', dpi=dpi, bbox_inches='tight'
    )  # TODO
    plt.close(fig)


def plot_valid_acc_alternet_curve():
    valid_max_subnet_csv_path = './data/csv/run-graduate_nb201_spos_altersubnet_exp1.3-tag-VAL_ACC_top1_val_acc_type_max.csv'
    x2, y2 = process_csv(valid_max_subnet_csv_path)

    valid_min_subnet_csv_path = './data/csv/run-graduate_nb201_spos_altersubnet_exp1.3-tag-VAL_ACC_top1_val_acc_type_min.csv'
    x4, y4 = process_csv(valid_min_subnet_csv_path)

    # figsize setting
    dpi = 100
    width, height = 1600, 1000
    legend_fontsize = 12
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)

    # plot range setting
    plt.xlim((0, 100))
    plt.ylim((60, 90))

    # plot grid background
    interval_x = 1000  # TODO
    interval_y = 5  # TODO
    plt.xticks(np.arange(0, max(x2) + interval_x, interval_x))
    plt.yticks(np.arange(60, 90 + interval_y, interval_y))
    plt.grid()

    # plot title and label
    # plt.title("EXP", fontsize=20)  # TODO
    plt.xlabel('The searching iter', fontsize=16)  # TODO
    plt.ylabel('The validation accuracy', fontsize=16)  # TODO

    # plot curve [linestyle: '-' or ':'] [color: 'g' or 'y']

    plt.plot(
        x2, y2, color='g', linestyle='-', label='Valid top1 accuracy (Max subnet)', lw=2
    )  # TODO
    plt.legend(loc=2, fontsize=legend_fontsize)  # TODO

    plt.plot(
        x4, y4, color='r', linestyle='-', label='Valid top1 accuracy (Min subnet)', lw=2
    )  # TODO
    plt.legend(loc=2, fontsize=legend_fontsize)  # TODO

    # save figure
    fig.savefig(
        './val_acc_alternet_chapt4_exp1.0.2.png', dpi=dpi, bbox_inches='tight'
    )  # TODO
    plt.close(fig)


if __name__ == '__main__':
    # plot_train_curve()
    # plot_valid_curve()
    # plot_valid_loss_alternet_curve()
    plot_valid_acc_alternet_curve()
