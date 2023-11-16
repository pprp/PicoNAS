import numpy as np

nb201_dict = np.load(
    '/data/lujunl/pprp/bench/nasbench201/nasbench201_dict.npy', allow_pickle=True
).item()

best_cifar10_valid_acc = 0.0
best_cifar10_test_acc = 0.0

best_cifar100_valid_acc = 0.0
best_cifar100_test_acc = 0.0

best_imagenet_valid_acc = 0.0
best_imagenet_test_acc = 0.0

for idx, _dict in nb201_dict.items():
    if _dict['cifar10_valid'] > best_cifar10_valid_acc:
        best_cifar10_valid_acc = _dict['cifar10_valid']

    if _dict['cifar10_test'] > best_cifar10_test_acc:
        best_cifar10_test_acc = _dict['cifar10_test']

    if _dict['cifar100_valid'] > best_cifar100_valid_acc:
        best_cifar100_valid_acc = _dict['cifar100_valid']

    if _dict['cifar100_test'] > best_cifar100_test_acc:
        best_cifar100_test_acc = _dict['cifar100_test']

    if _dict['imagenet16_valid'] > best_imagenet_valid_acc:
        best_imagenet_valid_acc = _dict['imagenet16_valid']

    if _dict['imagenet16_test'] > best_imagenet_test_acc:
        best_imagenet_test_acc = _dict['imagenet16_test']

print('best cifar10 valid acc: ', best_cifar10_valid_acc)
print('best cifar10 test acc: ', best_cifar10_test_acc)

print('best cifar100 valid acc: ', best_cifar100_valid_acc)
print('best cifar100 test acc: ', best_cifar100_test_acc)

print('best imagenet valid acc: ', best_imagenet_valid_acc)
print('best imagenet test acc: ', best_imagenet_test_acc)
