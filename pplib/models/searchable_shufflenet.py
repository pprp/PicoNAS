import numpy as np
import torch
import torch.nn as nn

from .modules import ShuffleModule, ShuffleXModule


class SinglePathOneShotSuperNet(nn.Module):

    def __init__(self,
                 dataset='cifar10',
                 input_size=32,
                 classes=10,
                 layers=20):
        super(SinglePathOneShotSuperNet, self).__init__()
        if dataset == 'cifar10':
            first_stride = 1
            self.downsample_layers = [4, 8]

        # ShuffleNet config
        self.last_channel = 1024
        self.channel = [
            16, 64, 64, 64, 64, 160, 160, 160, 160, 320, 320, 320, 320, 320,
            320, 320, 320, 640, 640, 640, 640
        ]

        self.kernel_list = [3, 5, 7, 'x']

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                3,
                self.channel[0],
                kernel_size=3,
                stride=first_stride,
                padding=1,
                bias=False),  # bias ?
            nn.BatchNorm2d(self.channel[0], affine=False),  # affine ?
            nn.ReLU6(inplace=True))

        self.choice_block = nn.ModuleList([])
        self.features = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inc, ouc = self.channel[i], self.channel[i + 1]
            else:
                stride = 1
                inc, ouc = self.channel[i] // 2, self.channel[i + 1]
            layer = nn.ModuleList([])
            for j in self.kernel_list:
                if j == 'x':
                    layer.append(ShuffleXModule(inc, ouc, stride=stride))
                else:
                    layer.append(
                        ShuffleModule(inc, ouc, kernel=j, stride=stride))
            self.features.append(layer)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                self.channel[-1], self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel, affine=False),
            nn.ReLU6(inplace=True))

        self.gvp = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.last_channel, classes, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, choice=np.random.randint(4, size=20)):
        # 20 = 4+4+8+4
        x = self.first_conv(input)
        for i, j in enumerate(choice):
            x = self.features[i][j](x)
        x = self.last_conv(x)
        x = self.gvp(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


class SinglePathOneShotSubNet(nn.Module):

    def __init__(self,
                 dataset='cifar10',
                 input_size=32,
                 classes=10,
                 layers=20,
                 choice=None):
        super(SinglePathOneShotSubNet, self).__init__()
        if dataset == 'cifar10':
            first_stride = 1
            self.downsample_layers = [4, 8]

        self.choice = np.random.randint(
            4, size=20) if choice is None else choice

        # ShuffleNet config
        self.last_channel = 1024
        self.channel = [
            16, 64, 64, 64, 64, 160, 160, 160, 160, 320, 320, 320, 320, 320,
            320, 320, 320, 640, 640, 640, 640
        ]

        self.kernel_list = [3, 5, 7, 'x']

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                3,
                self.channel[0],
                kernel_size=3,
                stride=first_stride,
                padding=1,
                bias=False),  # bias ?
            nn.BatchNorm2d(self.channel[0], affine=False),  # affine ?
            nn.ReLU6(inplace=True))

        self.choice_block = nn.ModuleList([])
        self.features = nn.ModuleList([])

        for i, c in enumerate(self.choice):
            if i in self.downsample_layers:
                stride = 2
                inc, ouc = self.channel[i], self.channel[i + 1]
            else:
                stride = 1
                inc, ouc = self.channel[i] // 2, self.channel[i + 1]
            layer = nn.ModuleList([])

            j = self.kernel_list[c]

            if j == 'x':
                layer.append(ShuffleXModule(inc, ouc, stride=stride))
            else:
                layer.append(ShuffleModule(inc, ouc, kernel=j, stride=stride))

            self.features.append(layer)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                self.channel[-1], self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel, affine=False),
            nn.ReLU6(inplace=True))

        self.gvp = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.last_channel, classes, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, choice):
        # 20 = 4+4+8+4
        x = self.first_conv(input)
        for i, j in enumerate(choice):
            x = self.features[i][j](x)
        x = self.last_conv(x)
        x = self.gvp(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    m = SinglePathOneShotSuperNet()
    input = torch.zeros(5, 3, 32, 32)
    print(m(input).shape)
