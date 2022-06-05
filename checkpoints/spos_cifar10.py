dataset_type = 'CIFAR10'
fast = False
nw = 2
data_dir = './data/cifar'
work_dir = './checkpoints/'
config = './configs/spos/spos_cifar10.py'
exp_name = 'spos_cifar10'
classes = 10
layers = 20
num_choices = 4
batch_size = 96
epochs = 600
learning_rate = 0.025
momentum = 0.9
weight_decay = 0.0003
val_interval = 5
random_search = 1000
dataset = 'cifar10'
cutout = False
cutout_length = 16
auto_aug = False
resize = False
