import torch
import torch.nn as nn
import torch.nn.functional as F


class Prediction(nn.Module):
    def __init__(self, num_channels):
        super(Prediction, self).__init__()
        self.num_layers = 4
        self.in_channel = num_channels
        self.kernel_size = 9
        self.num_filters = 64

        self.layer_in = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                  kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_in.weight.data)
        self.lam_in = nn.Parameter(torch.Tensor([0.01]))

        self.lam_i = []
        self.layer_down = []
        self.layer_up = []
        for i in range(self.num_layers):
            down_conv = 'down_conv_{}'.format(i)
            up_conv = 'up_conv_{}'.format(i)
            lam_id = 'lam_{}'.format(i)
            layer_2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_2.weight.data)
            setattr(self, down_conv, layer_2)
            self.layer_down.append(getattr(self, down_conv))
            layer_3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_3.weight.data)
            setattr(self, up_conv, layer_3)
            self.layer_up.append(getattr(self, up_conv))

            lam_ = nn.Parameter(torch.Tensor([0.01]))
            setattr(self, lam_id, lam_)
            self.lam_i.append(getattr(self, lam_id))

    def forward(self, mod):
        p1 = self.layer_in(mod)
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam_in))

        for i in range(self.num_layers):
            p3 = self.layer_down[i](tensor)
            p4 = self.layer_up[i](p3)
            p5 = tensor - p4
            p6 = torch.add(p1, p5)
            tensor = torch.mul(torch.sign(p6), F.relu(torch.abs(p6) - self.lam_i[i]))
        return tensor


class SFB(nn.Module):
    def __init__(self):
        super(SFB, self).__init__()
        self.channel = 64
        self.num_filters = 64
        self.kernel_size = 9
        self.net_u = Prediction(num_channels=self.channel)
        self.conv_u = nn.Conv2d(in_channels=self.num_filters, out_channels=64, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_u.weight.data)
        self.net_v = Prediction(num_channels=self.channel)
        self.conv_v = nn.Conv2d(in_channels=self.num_filters, out_channels=64, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_v.weight.data)
        self.net_z = Prediction(num_channels=2 * self.channel)
        self.relu_layer = nn.ReLU()
        self.norm = nn.BatchNorm2d(num_features=64)
        # self.decoder = decoder()

    def forward(self, x, y):

        u = self.net_u(x)
        v = self.net_v(y)

        p_x = self.conv_u(u)
        p_y = self.conv_v(v)

        p_x = self.relu_layer(p_x)
        # p_x = self.norm(p_x)
        p_y = self.relu_layer(p_y)
        # p_y = self.norm(p_y)

        # f_pred = torch.cat((p_x, p_y), dim=1)
        f_pred = p_y + p_x

        # z = self.net_z(p_xy)
        # f_pred = self.decoder(u,v,z)
        return f_pred