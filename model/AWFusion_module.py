import torch
import torch.nn as nn
import torchvision

from model.net import Decoder_simple
from model.net_lista import LRR_NET
from model.deweather_net import DeweatherNet
from utils22.img_read_save import RGB2YCrCb


class ColorResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ColorResidualBlock, self).__init__()

        # 第一个 3x3 卷积层 + BN + ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个 3x3 空洞卷积层 + BN + ReLU
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 第三个 3x3 卷积层 + BN
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class AWF(nn.Module):
    def __init__(self):
        super(AWF, self).__init__()
        self.didf_decoder = Decoder_simple(s=3, n=64, channel=1, stride=1, fusion_type='cat')
        self.lrr_net = LRR_NET(s=1, n=64, channel=1, stride=1, num_block=2)  # 12  4 2
        self.deweather_net = DeweatherNet()
        self.con1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.con2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.con3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, data_VIS_rgb, data_IR):
        # LRR_NET forward
        vi_ycbcr = RGB2YCrCb(data_VIS_rgb)
        data_VIS = vi_ycbcr[:, 0:1, :, :]

        V_L, V_S = self.lrr_net(data_VIS)
        I_L, I_S = self.lrr_net(data_IR)

        # DIDF_Decoder forward
        fuse = self.didf_decoder(V_L, I_L, V_S, I_S)
        rgb_Fuse = self.con1(fuse)
        rgb_IR = self.con2(data_IR)
        rgb_input = rgb_Fuse + data_VIS_rgb + rgb_IR
        final, I_Rtx, I_Rtx2, feature = self.deweather_net(rgb_input)

        return rgb_Fuse, final, I_Rtx, I_Rtx2, feature


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_VIS = torch.rand(1, 3, 256, 256).to(device)
    data_IR = torch.rand(1, 1, 256, 256).to(device)
    model = AWF().to(device)
    fuse, final, I_Rtx, I_Rtx2, cb_res, cr_res = model(data_VIS, data_IR)


