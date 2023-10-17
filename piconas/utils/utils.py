import argparse
import logging
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer as fvCheckpointer
from fvcore.common.config import CfgNode
from fvcore.common.file_io import PathManager
from scipy import stats
from sklearn import metrics
from torch.backends import cudnn

logger = logging.getLogger(__name__)


class Checkpointer(fvCheckpointer):

    def load(self,
             path: str,
             checkpointables: Optional[List[str]] = None) -> object:
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.
        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info(
                'No checkpoint found. Initializing model from scratch')
            return {}
        self.logger.info('Loading checkpoint from {}'.format(path))
        if not os.path.isfile(path):
            path = PathManager.get_local_path(path)
            assert os.path.isfile(path), 'Checkpoint {} not found!'.format(
                path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if incompatible is not None:
            # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:  # pyre-ignore
                self.logger.info('Loading {} from {}'.format(key, path))
                obj = self.checkpointables[key]
                try:
                    obj.load_state_dict(checkpoint.pop(key))  # pyre-ignore
                except:
                    print('exception loading')

        # return any further checkpoint data
        return checkpoint


class AverageMeter(object):

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


def count_parameters_in_MB(model):
    """
    Returns the model parameters in mega byte.
    """
    return (np.sum(
        np.prod(v.size())
        for name, v in model.named_parameters() if 'auxiliary' not in name) /
            1e6)


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
            metrics_dict['precision_{}'.format(k)] = sum(top_ytest
                                                         & top_test_pred) / k
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
    return f'Elapsed time: hour: {hour}, minute: {minute}, second: {second}'


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main
    path of residual blocks). This is the same as the DropConnect impl
    I created for EfficientNet, etc networks, however, the original name
    is misleading as 'Drop Connect' is a different form of dropout in a
    separate paper...
    See discussion:
        https://github.com/tensorflow/tpu/issues/494# issuecomment-532968956 ...
    I've opted for changing the layer and argument names to 'drop path'
    rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
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


def default_argument_parser():
    """
    Returns the argument parser with the default options.
    Inspired by the implementation of FAIR's detectron2
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, )
    parser.add_argument(
        '--config-file',
        default=None,
        metavar='FILE',
        help='Path to config file')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--datapath',
        default=None,
        metavar='FILE',
        help='Path to the folder with train/test data folders',
    )
    return parser


def parse_args(parser=default_argument_parser(), args=sys.argv[1:]):
    if '-f' in args:
        args = args[2:]
    return parser.parse_args(args)


def load_config(path):
    with open(path) as f:
        config = CfgNode.load_cfg(f)

    return config


def load_default_config():
    config_paths = 'configs/predictor_config.yaml'

    return load_config(config_paths)


def pairwise(iterable):
    """
    Iterate pairwise over list.
    from https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    """
    's -> (s0, s1), (s2, s3), (s4, s5), ...'
    a = iter(iterable)
    return zip(a, a)


def get_config_from_args(args=None):
    """
    Parses command line arguments and merges them with the defaults
    from the config file.
    Prepares experiment directories.
    Args:
        args: args from a different argument parser than the default one.
    """

    if args is None:
        args = parse_args()
    logger.info('Command line args: {}'.format(args))

    if args.config_file is None:
        config = load_default_config()
    else:
        config = load_config(path=args.config_file)

    # Override file args with ones from command line
    try:
        for arg, value in pairwise(args.opts):
            if '.' in arg:
                arg1, arg2 = arg.split('.')
                config[arg1][arg2] = type(config[arg1][arg2])(value)
            else:
                if arg in config:
                    t = type(config[arg])
                elif value.isnumeric():
                    t = int
                else:
                    t = str
                config[arg] = t(value)

        # load config file
        config.set_new_allowed(True)
        config.merge_from_list(args.opts)

    except AttributeError:
        for arg, value in pairwise(args):
            config[arg] = value

    if args.datapath is not None:
        config.train_data_file = os.path.join(args.datapath, 'train.json')
        config.test_data_file = os.path.join(args.datapath, 'test.json')
    else:
        config.train_data_file = None
        config.test_data_file = None

    # prepare the output directories
    config.save = '{}/{}/{}/{}/{}/{}'.format(
        config.out_dir,
        config.config_type,
        config.search_space,
        config.dataset,
        config.predictor,
        config.seed,
    )
    config.data = '{}/data'.format(get_project_root())

    create_exp_dir(config.save)
    create_exp_dir(config.save + '/search')  # required for the checkpoints
    create_exp_dir(config.save + '/eval')

    return config


def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent


def create_exp_dir(path):
    """
    Create the experiment directories.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger.info('Experiment dir : {}'.format(path))


def set_seed(seed):
    """
    Set the seeds for all used libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def log_args(args):
    """
    Log the args in a nice way.
    """
    for arg, val in args.items():
        logger.info(arg + '.' * (50 - len(arg) - len(str(val))) + str(val))


def accuracy_mse(predict, target, dataset, scale=100.):
    predict = dataset.denormalize(predict.detach()) * scale
    target = dataset.denormalize(target) * scale
    return F.mse_loss(predict, target)


def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_cuda(obj, device):
    if torch.is_tensor(obj):
        # return obj.cuda()
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_cuda(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_cuda(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_cuda(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


class AverageMeter(object):

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


class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeterV1(k, ':4f')
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return '  '.join(str(v) for v in self.meters.values())

    def summary(self):
        return '  '.join(v.summary() for v in self.meters.values())


class AverageMeterV1:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def time_record(start):
    import logging
    import time
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed Time: %dh %dmin %ds' % (hour, minute, second))


def gpu_monitor(gpu, sec, used=100):
    import logging
    import time

    import pynvml

    wait_min = sec // 60
    divisor = 1024 * 1024
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if meminfo.used / divisor < used:
        logging.info('GPU-{} is free, start runing!'.format(gpu))
        return False
    else:
        logging.info(
            'GPU-{}, Memory: total={}MB used={}MB free={}MB, waiting {}min...'.
            format(gpu, meminfo.total / divisor, meminfo.used / divisor,
                   meminfo.free / divisor, wait_min))
        time.sleep(sec)
        return True


def run_func(args, main):
    import time
    if torch.cuda.is_available():
        while gpu_monitor(args.gpu_id, sec=60, used=5000):
            pass
    start_time = time.time()
    result = main()
    time_record(start_time)
    # email_sender(result=result, config=args)
