import torch.nn as nn
import torch.nn.init as init
import torch


# ACNet
class ACBlock(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros", deploy=False, use_affine=True, reduce_gamma=False, gamma_init=None):
        super(ACBlock,self).__init__()
        self.deploy = deploy
        if self.deploy:
            self.fused_conv = nn.Conv2d(inc, ouc, kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            # main branch kxk
            self.square_conv = nn.Conv2d(inc, ouc,
                                         kernel_size=(
                                             kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(
                num_features=ouc, affine=use_affine)

            # padding calculation

            if padding - kernel_size // 2 < 0:
                raise "Not Comman Case: eg: K=3, P=1 or K=5, P=2"

            # common case
            self.crop = 0
            hor_padding = [padding - kernel_size // 2, padding]
            ver_padding = [padding, padding - kernel_size // 2]

            # kx1 conv
            self.ver_conv = nn.Conv2d(inc, ouc, kernel_size=(kernel_size, 1), stride=stride, padding=ver_padding,
                                      dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(ouc, affine=use_affine)

            # 1xk conv
            self.hor_conv = nn.Conv2d(inc, ouc, kernel_size=(1, kernel_size), stride=stride, padding=hor_padding,
                                      dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
            self.hor_bn = nn.BatchNorm2d(ouc, affine=use_affine)

            if reduce_gamma:
                self.init_gamma(1.0 / 3)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)

    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)

            ver_outputs = self.ver_conv(input)
            ver_outputs = self.ver_bn(ver_outputs)

            hor_outputs = self.hor_conv(input)
            hor_outputs = self.hor_bn(hor_outputs)

            return square_outputs + ver_outputs + hor_outputs

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        '''
        asymmetric kernel
        '''
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)

        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)

        square_kernel[:, :, 
                        square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                        square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(
            self.square_conv, self.square_bn)

        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        
        return square_k, hor_b + ver_b + square_b

    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels, out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size, stride=self.square_conv.stride,
                                    padding=self.square_conv.padding, dilation=self.square_conv.dilation, groups=self.square_conv.groups, bias=True,
                                    padding_mode=self.square_conv.padding_mode)
        self.__delattr__('square_conv')
        self.__delattr__('square_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b

if __name__ == '__main__':
    N = 1
    C = 2
    H = 62
    W = 62
    O = 8
    groups = 4

    x = torch.randn(N, C, H, W)
    print('input shape is ', x.size())

    test_kernel_padding = [(3,1),  (5,2)]

    for k, p in test_kernel_padding:
        acb = ACBlock(C, O, kernel_size=k, padding=p, stride=1, deploy=False)
        acb.eval()
        for module in acb.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.uniform_(module.running_mean, 0, 0.1)
                nn.init.uniform_(module.running_var, 0, 0.2)
                nn.init.uniform_(module.weight, 0, 0.3)
                nn.init.uniform_(module.bias, 0, 0.4)
        out = acb(x)
        acb.switch_to_deploy()
        deployout = acb(x)
        print('difference between the outputs of the training-time and converted ACB is')
        print(((deployout - out) ** 2).sum())