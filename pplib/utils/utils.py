import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn import metrics

logger = logging.getLogger(__name__)


def compute_scores(ytest, test_pred):
    ytest = np.array(ytest)
    test_pred = np.array(test_pred)
    METRICS = [
        'mae',
        'rmse',
        'pearson',
        'spearman',
        'kendalltau',
        'kt_2dec',
        'kt_1dec',
        'full_ytest',
        'full_testpred',
    ]
    metrics_dict = {}

    try:
        metrics_dict['mae'] = np.mean(abs(test_pred - ytest))
        metrics_dict['rmse'] = metrics.mean_squared_error(
            ytest, test_pred, squared=False)
        metrics_dict['pearson'] = np.abs(np.corrcoef(ytest, test_pred)[1, 0])
        metrics_dict['spearman'] = stats.spearmanr(ytest, test_pred)[0]
        metrics_dict['kendalltau'] = stats.kendalltau(ytest, test_pred)[0]
        metrics_dict['kt_2dec'] = stats.kendalltau(
            ytest, np.round(test_pred, decimals=2))[0]
        metrics_dict['kt_1dec'] = stats.kendalltau(
            ytest, np.round(test_pred, decimals=1))[0]
        for k in [10, 20]:
            top_ytest = np.array(
                [y > sorted(ytest)[max(-len(ytest), -k - 1)] for y in ytest])
            top_test_pred = np.array([
                y > sorted(test_pred)[max(-len(test_pred), -k - 1)]
                for y in test_pred
            ])
            metrics_dict['precision_{}'.format(k)] = (
                sum(top_ytest & top_test_pred) / k)
        metrics_dict['full_ytest'] = ytest.tolist()
        metrics_dict['full_testpred'] = test_pred.tolist()

    except:
        for metric in METRICS:
            metrics_dict[metric] = float('nan')
    if np.isnan(metrics_dict['pearson']) or not np.isfinite(
            metrics_dict['pearson']):
        logger.info('Error when computing metrics. ytest and test_pred are:')
        logger.info(ytest)
        logger.info(test_pred)

    return metrics_dict


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1, )):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, exp_name=None, iters=0, tag=''):
    if exp_name is not None:
        exp_full_path = os.path.join('./checkpoints', exp_name)
    else:
        exp_full_path = './checkpoints'
    if not os.path.exists(exp_full_path):
        os.makedirs(exp_full_path)
    filename = os.path.join(exp_full_path,
                            '{}_ckpt_{:04}.pth.tar'.format(tag, iters))
    torch.save(state, filename)


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('Elapsed time: hour: %d, minute: %d, second: %f' %
          (hour, minute, second))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main
    path of residual blocks). This is the same as the DropConnect impl
    I created for EfficientNet, etc networks, however, the original name
    is misleading as 'Drop Connect' is a different form of dropout in a
    separate paper...
    See discussion:
        https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    I've opted for changing the layer and argument names to 'drop path'
    rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample

    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def iter_flatten(iterable):
    """
    Flatten a potentially deeply nested python list
    """
    # https://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            yield from iter_flatten(e)
        else:
            yield e


class AttrDict(dict):
    """Convert the key of dict to attribute of objects"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
