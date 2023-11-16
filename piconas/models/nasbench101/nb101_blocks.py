import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, inplanes, outplanes, k):
        super(ConvBnRelu, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),
            nn.Conv2d(
                outplanes,
                outplanes,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                bias=False,
            ),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.op(x)


class MaxPool(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(MaxPool, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),
            nn.MaxPool2d(3, 1, padding=1),
        )

    def forward(self, x):
        return self.op(x)
