import numpy as np
from tensorflow.python.keras.utils import to_categorical
import copy
from collections import namedtuple, OrderedDict
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
import os
import shutil
import logging
import sys

PADDING_MAX_LENGTH = 9


# delete the first column and the last row in the matrix (because they are zeros)
# input: adjacent matrix (9*9)
# output: adjacent matrix (8*8)
def delete_margin(matrix):
    return matrix[:-1, 1:]


# this function is for padding zero for matrix which is not 9*9
# zeros will be added at penultimate row (or column)
# input: important_metrics
# output: the metrics after padding
def padding_zero_in_matrix(important_metrics, max_length=PADDING_MAX_LENGTH):
    for i in important_metrics:
        len_operations = len(important_metrics[i]['fixed_metrics']['module_operations'])
        if len_operations != max_length:
            # if the operations is less than max_length
            for j in range(len_operations, max_length):
                important_metrics[i]['fixed_metrics']['module_operations'].insert(-1, 'null')
            # print(important_metrics[i]['fixed_metrics']['module_operations'])

            adjecent_matrix = important_metrics[i]['fixed_metrics']['module_adjacency']
            padding_matrix = np.insert(adjecent_matrix, len_operations - 1,
                                       np.zeros([max_length - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1],
                                       np.zeros([max_length, max_length - len_operations]), axis=1)
            important_metrics[i]['fixed_metrics']['module_adjacency'] = padding_matrix
    return important_metrics


def padding_zeros(matrix, op_list, max_length=PADDING_MAX_LENGTH):
    assert len(op_list) == len(matrix)
    padding_matrix = matrix
    len_operations = len(op_list)
    if not len_operations == max_length:
        for j in range(len_operations, max_length):
            op_list.insert(j - 1, 'null')
        adjecent_matrix = copy.deepcopy(matrix)
        padding_matrix = np.insert(adjecent_matrix, len_operations - 1,
                                   np.zeros([max_length - len_operations, len_operations]),
                                   axis=0)
        padding_matrix = np.insert(padding_matrix, [len_operations - 1],
                                   np.zeros([max_length, max_length - len_operations]), axis=1)

    return padding_matrix, op_list


def padding_zeros_darts(matrixes, ops, max_length=PADDING_MAX_LENGTH):
    '''
    input darts cell matrix and ops
    :param matrixes: len=2
    :param ops: len=2
    :return: the matrix and ops after padding
    '''
    padding_matrixes = []
    padding_ops = []
    for matrix, op in zip(matrixes, ops):
        if op is None:
            # matrix is None this case
            padding_matrix = np.zeros(shape=[max_length, max_length], dtype='int8')
            tmp_op = np.zeros(shape=max_length, dtype='int8')

            padding_matrixes.append(padding_matrix)
            padding_ops.append(tmp_op)
            continue

        len_operations = len(op)
        tmp_op = copy.deepcopy(op)
        padding_matrix = copy.deepcopy(matrix)
        if not len_operations == max_length:
            for j in range(len_operations, max_length):
                tmp_op.insert(j - 1, 0)

            padding_matrix = np.insert(padding_matrix, len_operations - 1,
                                       np.zeros([max_length - len_operations, len_operations]),
                                       axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1],
                                       np.zeros([max_length, max_length - len_operations]), axis=1)
        padding_matrixes.append(padding_matrix)
        padding_ops.append(tmp_op)
    return padding_matrixes, padding_ops


# return: 1. X: a linear array: flattened adjacent matrix + integer operations
#         2. y: accuracy
def get_bit_data(important_metrics, integers2one_hot=True):
    X = []
    y = []
    for index in important_metrics:
        fixed_metrics = important_metrics[index]['fixed_metrics']
        adjacent_matrix = fixed_metrics['module_adjacency']
        module_integers = fixed_metrics['module_integers']
        accuracy = important_metrics[index]['final_valid_accuracy']

        adjacent_matrix = delete_margin(adjacent_matrix)
        flattened_adjacent = adjacent_matrix.flatten()
        input_metrics = []
        input_metrics.extend(flattened_adjacent)
        if integers2one_hot:
            module_integers = to_categorical(module_integers, 4, dtype='int8')
            module_integers = module_integers.flatten()
        input_metrics.extend(module_integers)
        X.append(input_metrics)
        y.append(accuracy)

    assert len(X) == len(y)

    return X, y


def get_bit_data_darts(important_metrics, integers2one_hot=True):
    X = []
    # each data in X consist with 4 dim rows
    for index in important_metrics:
        fixed_metrics = important_metrics[index]
        padding_norm_matrixes = fixed_metrics['padding_norm_matrixes']
        padding_norm_ops = fixed_metrics['padding_norm_ops']
        padding_reduc_matrixes = fixed_metrics['padding_reduc_matrixes']
        padding_reduc_ops = fixed_metrics['padding_reduc_ops']
        matrixes = padding_norm_matrixes + padding_reduc_matrixes
        ops = padding_norm_ops + padding_reduc_ops
        assert len(matrixes) == 4
        assert len(ops) == 4
        each_x = []  # len(each_x)==4
        for adjacency_matrix, op in zip(matrixes, ops):
            adjacency_matrix = delete_margin(adjacency_matrix)
            flattened_adjacency = adjacency_matrix.flatten()
            input_x = []
            input_x.extend(flattened_adjacency)
            if integers2one_hot:
                op = to_categorical(op, 4, dtype='int8')
                op = op.flatten()
            input_x.extend(op)
            each_x.append(input_x)
        X.append(each_x)

    return X

def get_matrix_data_darts(important_metrics):
    m, o = [], []
    for index in important_metrics:
        fixed_metrics = important_metrics[index]
        padding_norm_matrixes = fixed_metrics['padding_norm_matrixes']
        padding_norm_ops = fixed_metrics['padding_norm_ops']
        padding_reduc_matrixes = fixed_metrics['padding_reduc_matrixes']
        padding_reduc_ops = fixed_metrics['padding_reduc_ops']
        matrixes = padding_norm_matrixes + padding_reduc_matrixes
        ops = padding_norm_ops + padding_reduc_ops
        assert len(matrixes) == 4
        assert len(ops) == 4
        m.append(matrixes)
        o.append(ops)

    return m, o


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def convert_to_genotype(arch):
    op_dict = {
        0: 'none',
        1: 'sep_conv_5x5',
        2: 'dil_conv_5x5',
        3: 'sep_conv_3x3',
        4: 'dil_conv_3x3',
        5: 'max_pool_3x3',
        6: 'avg_pool_3x3',
        7: 'skip_connect'
    }
    darts_arch = [[], []]
    i = 0
    for cell in arch:
        for n in cell:
            darts_arch[i].append((op_dict[n[1]], n[0]))
        i += 1
    geno = Genotype(normal=darts_arch[0], normal_concat=[2, 3, 4, 5], reduce=darts_arch[1], reduce_concat=[2, 3, 4, 5])
    return geno


# below is from DARTS code
def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


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


class AverageMeter:
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

class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def get_logger():
    time_format = "%m/%d %H:%M:%S"
    fmt = "[%(asctime)s] %(levelname)s (%(name)s) %(message)s"
    formatter = logging.Formatter(fmt, time_format)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    if isinstance(obj, tuple):
        return tuple(to_cuda(t) for t in obj)
    if isinstance(obj, list):
        return [to_cuda(t) for t in obj]
    if isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))