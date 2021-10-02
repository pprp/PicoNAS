import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules import padding


def channel_shuffle(x):
    """
        code from https://github.com/megvii-model/SinglePathOneShot/src/Search/blocks.py#L124
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class ShuffleModule(nn.Module):
    def __init__(self, inc, ouc, kernel, stride, supernet=True):
        super(ShuffleModule, self).__init__()
        self.affine = supernet
        self.stride = stride
        self.kernel = kernel
        self.inc = inc
        self.ouc = ouc - inc
        self.midc = ouc // 2
        self.padding = kernel // 2

        self.branch = nn.Sequential(
            # point wise conv2d
            nn.Conv2d(self.inc, self.midc, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            nn.ReLU(inplace=True),

            # depth wise conv2d
            nn.Conv2d(self.midc, self.midc, kernel_size=self.kernel,
                      stride=self.stride, padding=self.padding, bias=False, groups=self.midc),
            nn.BatchNorm2d(self.midc, affine=self.affine),

            # point wise conv2d
            nn.Conv2d(self.midc, self.ouc, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ouc, affine=self.affine),
            nn.ReLU(inplace=True)
        )

        if self.stride == 2:
            self.proj = nn.Sequential(
                # depth wise conv2d
                nn.Conv2d(self.inc, self.inc, kernel_size=self.kernel, stride=2,
                          padding=self.padding, bias=False, groups=self.inc),
                nn.BatchNorm2d(self.inc, affine=self.affine),

                # point wise conv2d
                nn.Conv2d(self.inc, self.inc, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inc, affine=self.affine),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle(x)
            y = torch.cat((self.branch(x1), x2), 1)
        else:
            y = torch.cat((self.branch(x), self.proj(x)), 1)
        return y


class ShuffleXModule(nn.Module):
    def __init__(self, inc, ouc, stride, supernet=True):
        super(ShuffleXModule, self).__init__()
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.inc = inc
        self.midc = ouc // 2
        self.ouc = ouc - inc

        self.cb_main = nn.Sequential(
            # dw
            nn.Conv2d(self.inc, self.inc, kernel_size=3, stride=stride,
                      padding=1, bias=False, groups=self.inc),
            nn.BatchNorm2d(self.inc, affine=self.affine),
            # pw
            nn.Conv2d(self.inc, self.midc, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.midc, self.midc, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.midc),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            # pw
            nn.Conv2d(self.midc, self.midc, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.midc, self.midc, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.midc),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            # pw
            nn.Conv2d(self.midc, self.ouc, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ouc, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.inc, self.inc, kernel_size=3, stride=2,
                          padding=1, groups=self.inc, bias=False),
                nn.BatchNorm2d(self.inc, affine=self.affine),
                # pw
                nn.Conv2d(self.inc, self.inc, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inc, affine=self.affine),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle(x)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y


class SinglePathOneShotCIFAR(nn.Module):
    def __init__(self, dataset='cifar10', input_size=32, classes=10, layers=20):
        super(SinglePathOneShotCIFAR, self).__init__()
        if dataset == 'cifar10':
            first_stride = 1
            self.downsample_layers = [4, 8]

        # ShuffleNet config
        self.last_channel = 1024
        self.channel = [16,
                        64, 64, 64, 64,
                        160, 160, 160, 160,
                        320, 320, 320, 320, 320, 320, 320, 320,
                        640, 640, 640, 640]

        self.kernel_list = [3, 5, 7, 'x']

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, self.channel[0], kernel_size=3,
                      stride=first_stride, padding=1, bias=False),  # bias ?
            nn.BatchNorm2d(self.channel[0], affine=False),  # affine ?
            nn.ReLU6(inplace=True)
        )

        self.choice_block = nn.ModuleList([])
        self.features = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inc, ouc = self.channel[i], self.channel[i+1]
            else:
                stride = 1
                inc, ouc = self.channel[i] // 2, self.channel[i+1]
            layer = nn.ModuleList([])
            for j in self.kernel_list:
                if j == 'x':
                    layer.append(ShuffleXModule(inc, ouc, stride=stride))
                else:
                    layer.append(ShuffleModule(
                        inc, ouc, kernel=j, stride=stride))
            self.features.append(layer)

        self.last_conv = nn.Sequential(
            nn.Conv2d(self.channel[-1],
                      self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel, affine=False),
            nn.ReLU6(inplace=True)
        )

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
        print(x.shape)
        for i, j in enumerate(choice):
            x = self.features[i][j](x)
            print(x.shape)
        x = self.last_conv(x)
        print(x.shape)
        x = self.gvp(x)
        x = x.view(-1, self.last_channel)
        print(x.shape)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    m = SinglePathOneShotCIFAR()
    input = torch.zeros(5, 3, 32, 32)
    print(m(input).shape)
