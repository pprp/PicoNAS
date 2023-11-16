import torch
import torch.nn as nn


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % 4 == 0
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
            nn.Conv2d(
                self.inc, self.midc, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            nn.ReLU(inplace=True),
            # depth wise conv2d
            nn.Conv2d(
                self.midc,
                self.midc,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
                bias=False,
                groups=self.midc,
            ),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            # point wise conv2d
            nn.Conv2d(
                self.midc, self.ouc, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.ouc, affine=self.affine),
            nn.ReLU(inplace=True),
        )

        if self.stride == 2:
            self.proj = nn.Sequential(
                # depth wise conv2d
                nn.Conv2d(
                    self.inc,
                    self.inc,
                    kernel_size=self.kernel,
                    stride=2,
                    padding=self.padding,
                    bias=False,
                    groups=self.inc,
                ),
                nn.BatchNorm2d(self.inc, affine=self.affine),
                # point wise conv2d
                nn.Conv2d(
                    self.inc, self.inc, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(self.inc, affine=self.affine),
                nn.ReLU(inplace=True),
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
            nn.Conv2d(
                self.inc,
                self.inc,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                groups=self.inc,
            ),
            nn.BatchNorm2d(self.inc, affine=self.affine),
            # pw
            nn.Conv2d(
                self.inc, self.midc, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                self.midc,
                self.midc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=self.midc,
            ),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            # pw
            nn.Conv2d(
                self.midc, self.midc, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                self.midc,
                self.midc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=self.midc,
            ),
            nn.BatchNorm2d(self.midc, affine=self.affine),
            # pw
            nn.Conv2d(
                self.midc, self.ouc, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.ouc, affine=self.affine),
            nn.ReLU(inplace=True),
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(
                    self.inc,
                    self.inc,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.inc,
                    bias=False,
                ),
                nn.BatchNorm2d(self.inc, affine=self.affine),
                # pw
                nn.Conv2d(
                    self.inc, self.inc, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(self.inc, affine=self.affine),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle(x)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y


blocks_dict = {
    'mobilenet_3x3_ratio_3': lambda inp, oup, stride: InvertedResidual(
        inp, oup, 3, 1, stride, 3
    ),
    'mobilenet_3x3_ratio_6': lambda inp, oup, stride: InvertedResidual(
        inp, oup, 3, 1, stride, 6
    ),
    'mobilenet_5x5_ratio_3': lambda inp, oup, stride: InvertedResidual(
        inp, oup, 5, 2, stride, 3
    ),
    'mobilenet_5x5_ratio_6': lambda inp, oup, stride: InvertedResidual(
        inp, oup, 5, 2, stride, 6
    ),
    'mobilenet_7x7_ratio_3': lambda inp, oup, stride: InvertedResidual(
        inp, oup, 7, 3, stride, 3
    ),
    'mobilenet_7x7_ratio_6': lambda inp, oup, stride: InvertedResidual(
        inp, oup, 7, 3, stride, 6
    ),
}


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, ksize, padding, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, ksize, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    inp * expand_ratio,
                    inp * expand_ratio,
                    ksize,
                    stride,
                    padding,
                    groups=inp * expand_ratio,
                    bias=False,
                ),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
