# draw line with
import numpy as np


def plot_curve(self, save_path):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    title = 'the accuracy/loss curve of train/val'
    dpi = 100
    width, height = 1600, 1000
    legend_fontsize = 10
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(
        x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(
        x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(
        x_axis,
        y_axis * 50,
        color='g',
        linestyle=':',
        label='train-loss-x50',
        lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(
        x_axis,
        y_axis * 50,
        color='y',
        linestyle=':',
        label='valid-loss-x50',
        lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)
