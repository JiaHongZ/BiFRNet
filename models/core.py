import torch
from torch import nn
from torch.nn import functional as F
from models.conv_lstm import Conv2dLSTMCell

class InferenceCore(nn.Module):
    def __init__(self):
        super(InferenceCore, self).__init__()
        self.downsample_x = nn.Conv2d(3, 3, kernel_size=4, stride=4, padding=0, bias=False)
        self.upsample_v = nn.ConvTranspose2d(3, 3, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256, 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        self.core = Conv2dLSTMCell(3+3+256+2*128, 128, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x, v, r, c_e, h_e, h_g, u):
        x = self.downsample_x(x)
        v = self.upsample_v(v.view(-1, 3, 1, 1))
        if r.size(2)!=h_e.size(2):
            r = self.upsample_r(r)
        u = self.downsample_u(u)
        c_e, h_e = self.core(torch.cat((x, v, r, h_g, u), dim=1), (c_e, h_e))
        
        return c_e, h_e
    
class GenerationCore(nn.Module):
    def __init__(self):
        super(GenerationCore, self).__init__()
        self.upsample_v = nn.ConvTranspose2d(3, 3, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256, 256, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(3+256+3, 128, kernel_size=5, stride=1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)
        
    def forward(self, v, r, c_g, h_g, u, z):
        v = self.upsample_v(v.view(-1, 3, 1, 1))
        if r.size(2)!=h_g.size(2):
            r = self.upsample_r(r)
        c_g, h_g =  self.core(torch.cat((v, r, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u
        
        return c_g, h_g, u


class GenerationCoreK1(nn.Module):
    def __init__(self):
        super(GenerationCoreK1, self).__init__()
        self.conv_v = nn.Conv2d(512*12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_r = nn.Conv2d(512*12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, c_g, h_g, u):
        b,c,h,w = c_g.shape
        c_g = self.conv_v(c_g)
        h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((u,c_g), dim=1), (c_g, h_g))
        u = self.conv(h_g)

        return c_g, h_g, u
class GenerationCoreK2(nn.Module):
    def __init__(self):
        super(GenerationCoreK2, self).__init__()
        self.conv_v = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, c_g, h_g, u):
        b,c,h,w = c_g.shape
        c_g = self.conv_v(c_g)
        h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((u,c_g), dim=1), (c_g, h_g))
        u = self.conv(h_g)

        return c_g, h_g, u
# v1 or v2
class GenerationCoreK_v1orv2(nn.Module):
    def __init__(self):
        super(GenerationCoreK_v1orv2, self).__init__()
        self.conv_v = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, c_g, h_g, u):
        b,c,h,w = c_g.shape
        c_g = self.conv_v(c_g)
        h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((u,c_g), dim=1), (c_g, h_g))
        u = self.conv(h_g) + u

        return c_g, h_g, u



class GenerationCoreK4(nn.Module):
    def __init__(self):
        super(GenerationCoreK4, self).__init__()
        self.conv_x = nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_v = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12*2, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x_q, c_g, h_g, u):
        b,c,h,w = c_g.shape
        x_q = self.conv_x(x_q)
        c_g = self.conv_v(c_g)
        h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((u,x_q,h_g), dim=1), (c_g, h_g))
        u = self.conv(c_g) + u

        return c_g, h_g, u

class GenerationCoreK3(nn.Module):
    def __init__(self):
        super(GenerationCoreK, self).__init__()
        self.conv_v = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, c_g, h_g, u):
        b,c,h,w = c_g.shape
        c_g = self.conv_v(c_g)
        h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((u,h_g), dim=1), (c_g, h_g))
        u = self.conv(c_g) + u

        return c_g, h_g, u

class GenerationCoreKv5(nn.Module):
    def __init__(self):
        super(GenerationCoreK, self).__init__()
        self.conv_x = nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_v = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x_q, c_g, h_g, u):
        b,c,h,w = c_g.shape
        x_q = self.conv_x(x_q)
        c_g = self.conv_v(c_g)
        h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((u,x_q), dim=1), (c_g, h_g))
        u = self.conv(c_g) + u
        # u = self.conv(c_g)

        return c_g, h_g, u

# class GenerationCoreK(nn.Module):
#     def __init__(self):
#         super(GenerationCoreK, self).__init__()
#         self.conv_v = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
#         self.core = Conv2dLSTMCell(512+12, 12, kernel_size=5, stride=1, padding=2)
#         self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)
#
#     def forward(self, c_g, h_g, u):
#         b,c,h,w = c_g.shape
#         c_g = self.conv_v(c_g)
#         h_g = self.conv_r(h_g)
#         c_g, h_g = self.core(torch.cat((u,c_g), dim=1), (c_g, h_g))
#         u = self.conv(h_g) + u
#
#         return c_g, h_g, u
class GenerationCoreKv7(nn.Module):
    def __init__(self):
        super(GenerationCoreK, self).__init__()
        self.conv_x = nn.Conv2d(512, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_v = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)
    def forward(self, x_q, c_g, h_g, u):
        b,c,h,w = c_g.shape
        x_q = self.conv_x(x_q)
        c_g = self.conv_v(c_g)
        h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((u,x_q), dim=1), (c_g, h_g))
        u = self.conv(c_g) + u
        return c_g, h_g, u


class GenerationCoreK_v8(nn.Module):
    def __init__(self):
        super(GenerationCoreK_v8, self).__init__()
        self.conv_x = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)
    def forward(self, x_q, z, c_g, h_g, u):
        x_q = self.conv_x(x_q)
        h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((x_q,z), dim=1), (c_g, h_g))
        u = self.conv(h_g) + u
        return c_g, h_g, u

class GenerationCoreK_v9(nn.Module):
    def __init__(self):
        super(GenerationCoreK_v9, self).__init__()
        self.conv_x = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512+12+12, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)
    def forward(self, x_q, z1, z2, c_g, h_g, u):
        x_q = self.conv_x(x_q)
        # h_g = self.conv_r(h_g)
        c_g, h_g = self.core(torch.cat((x_q,z1,z2), dim=1), (c_g, h_g))
        u = self.conv(h_g) + u
        return c_g, h_g, u

# 最好的
class GenerationCoreK(nn.Module):
    def __init__(self):
        super(GenerationCoreK, self).__init__()
        self.conv_x = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)
    def forward(self, x_q, c_g, h_g, u):
        x_q = self.conv_x(x_q)
        # h_g = self.conv_r(h_g)
        c_g, h_g = self.core(x_q, (c_g, h_g))
        u = self.conv(h_g) + u
        return c_g, h_g, u

# class GenerationCoreK(nn.Module):
#     def __init__(self):
#         super(GenerationCoreK, self).__init__()
#         self.conv_x = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
#         # self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
#         self.core = Conv2dLSTMCell(512, 12*32, kernel_size=5, stride=1, padding=2)
#         self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)
#     def forward(self, x_q, c_g, h_g, u):
#         x_q = self.conv_x(x_q)
#         # h_g = self.conv_r(h_g)
#         c_g, h_g = self.core(x_q, (c_g, h_g))
#         u = self.conv(h_g) + u
#         return c_g, h_g, u

class GenerationCoreK_knowledgedonotwork(nn.Module):
    def __init__(self):
        super(GenerationCoreK, self).__init__()
        self.conv_x = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_r = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(512, 12, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(12, 512, kernel_size=3, stride=1, padding=1, bias=True)
    def forward(self, x_q, c_g, h_g, u):
        x_q = self.conv_x(x_q)
        # h_g = self.conv_r(h_g)
        c_g, h_g = self.core(x_q, (c_g, h_g))
        u = self.conv(h_g) + u
        return c_g, h_g, u