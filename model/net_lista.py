import torch
import torch.nn as nn
import numpy as np

class LRR_NET(nn.Module):
    def __init__(self, s, n, channel, stride, num_block,):
        super(LRR_NET, self).__init__()
        self.get_ls1 = GetLS_Net(s, n, channel, stride, num_block)
        # self.get_ls2 = GetLS_Net(s, n, channel, stride, num_block)

    def forward(self, x):
        fea_l, fea_s = self.get_ls1(x)
        # fea_y_l, fea_y_s = self.get_ls2(y)

        return fea_l, fea_s

class GetLS_Net(nn.Module):
    def __init__(self, s, n, channel, stride, num_block):
        super(GetLS_Net, self).__init__()
        # n = 128  # number of filters
        # s = 3  # filter size
        # num_block = 4  # number of layers
        # Channel = 3
        self.n = n
        self.num_block = num_block
        self.conv_W00 = ConvLayer(channel, 2*n, s, stride)
        self.lamj = nn.Parameter(torch.rand(1, self.n*2))  # l1-norm
        self.lamz = nn.Parameter(torch.rand(1, 1))
        self.up = nn.Upsample(scale_factor=2)
        self.norm = nn.BatchNorm2d(num_features=2*n)
        for i in range(num_block):
            self.add_module('lrrblock' + str(i), LRR_Block_lista(s, 2 * n, channel, stride))

    def forward(self, x):
        b, c, h, w = x.shape
        tensor_l = self.conv_W00(x)  # Z_0
        tensor_zz = eta_l1(tensor_l, self.lamj)
        # tensor_zz = self.norm(tensor_z)

        for i in range(self.num_block):
            # print('num_block - ' + str(i))
            lrrblock = getattr(self, 'lrrblock' + str(i))
            tensor_zz = lrrblock(x, tensor_zz, self.lamj, self.lamz)
        L = tensor_zz[:, :self.n, :, :]
        S = tensor_zz[:, self.n: 2 * self.n, :, :]
        return L, S

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        # print(out.dtype)
        out = self.conv2d(out)
        return out

class LRR_Block_lista(nn.Module):
    def __init__(self, s, n, c, stride):
        super(LRR_Block_lista, self).__init__()
        self.conv_Wdz = ConvLayer(n, c, s, stride)
        self.conv_Wdtz = ConvLayer(c, n, s, stride)

    def forward(self, x, tensor_z, lam_theta, lam_z):
        # Updating
        convZ1 = self.conv_Wdz(tensor_z)
        midZ = x - convZ1
        tensor_c = lam_z*tensor_z + self.conv_Wdtz(midZ)
        # tensor_c = tensor_b + hZ
        Z = eta_l1(tensor_c, lam_theta)
        return Z

def eta_l1(r_, lam_):
    # l_1 norm based
    # implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)
    B, C, H, W = r_.shape
    lam__ = torch.reshape(lam_, [1, C, 1, 1])
    lam___ = lam__.repeat(B, 1, H, W)
    R = torch.sign(r_) * torch.clamp(torch.abs(r_) - lam___, 0)
    return R