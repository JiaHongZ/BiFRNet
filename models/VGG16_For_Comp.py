from einops.layers.torch import Rearrange
from typing import Optional
import torch
from torch import nn
from einops import rearrange, repeat
from torchvision import models
import numpy as np
import os
from einops import rearrange, repeat

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
class Vgg16_1(torch.nn.Module):
    """
    Used in the original NST paper, only those layers are exposed which were used in the original paper
    """
    def __init__(self):
        super().__init__()
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        # self.vgg_pretrained_avgpool = models.vgg16(pretrained=True).avgpool
        # print(vgg_pretrained_features)
        # 冻结参数

        # 添加后面的        # for param in self.vgg_pretrained_features.parameters():
        #     param.requires_grad = True
        # for param in self.vgg_pretrained_avgpool.parameters():
        #     param.requires_grad = True层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 12),
        )
    def forward(self, x):
        x = self.vgg_pretrained_features(x)
        # x = self.vgg_pretrained_avgpool(x)
        y = torch.flatten(x, 1)
        y = self.classifier(y)
        return y
class Vgg16(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True, progress=show_progress).features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.content_feature_maps_index = 1  # relu2_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))  # all layers used for style representation

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(5):
            self.slice1.add_module(str(x), vgg_pretrained_features[x]) # 112
        for x in range(5, 10):
            self.slice2.add_module(str(x), vgg_pretrained_features[x]) # 56
        for x in range(10, 17):
            self.slice3.add_module(str(x), vgg_pretrained_features[x]) # 28
        for x in range(17, 24):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(24, 31):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, x):
        x1 = self.slice1(x) # 112
        x2 = self.slice2(x1) # 56
        x3 = self.slice3(x2) # 28
        x4 = self.slice4(x3) # 14
        x5 = self.slice5(x4) # 7
        return x1,x2,x3,x4,x5
class Vgg16_classifier(torch.nn.Module):
    """
    Used in the original NST paper, only those layers are exposed which were used in the original paper
    """
    def __init__(self, encoder_dim, classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classes),
        )
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_dim=None):
        super().__init__()

        mlp_dim = dim if mlp_dim is None else mlp_dim
        self.linear_1 = nn.Linear(dim, mlp_dim)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(mlp_dim, mlp_dim)

    def forward(self, x):
        x = self.linear_1(x)  # (b, *, mlp_dim)
        x = self.activation(x)  # (b, *, mlp_dim)
        x = self.linear_2(x)  # (b, *, dim)
        return x
class SAM(nn.Module):
    """
    SAM Network.
    """
    def __init__(self, input_channel, hidden, image_size, decoder_dim, SAM_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param SAM_dim: size of the SAM network
        """
        super(SAM, self).__init__()
        # 特征提取
        # self.features = Vgg16_extractor(hiddens=hidden,image_size=image_size) # 7x7
        self.features = Vgg16() # 7x7
        self.xl1 = nn.Sequential(
            nn.Conv2d(64,32, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32, kernel_size=1, stride=1, padding=0),
        )
        self.xl2 = nn.Sequential(
            nn.Conv2d(160,32, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32, kernel_size=1, stride=1, padding=0),
        )
        self.xl3 = nn.Sequential(
            nn.Conv2d(288,32, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32, kernel_size=1, stride=1, padding=0),
        )
        self.xl4 = nn.Sequential(
            nn.Conv2d(544,32, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32, kernel_size=1, stride=1, padding=0),
        )
        self.xl5 = nn.Sequential(
            nn.Conv2d(544,32, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.Conv2d(32,1, kernel_size=1, stride=1, padding=0),
        )

        self.f_beta = nn.Linear(hidden*image_size*image_size, image_size*image_size)  # linear layer to create a sigmoid-activated gate
        self.ln = nn.LayerNorm(7*7)
        # self.softmax = nn.Softmax()  # softmax layer to calculate weights
        self.sigmoid = nn.Sigmoid()

        self.att_weight = None # 记录
        # self.conv = nn.Conv2d(hidden, 64, 3, 1, 1)

    def forward(self, x):
        x1,x2,x3,x4,x5 = self.features(x) # b 512 7 7
        f_xl1 = self.xl1(x1)
        f_xl1 = torch.cat([f_xl1,x2],1)
        f_xl2 = self.xl2(f_xl1)
        f_xl2 = torch.cat([f_xl2,x3],1)
        f_xl3 = self.xl3(f_xl2)
        f_xl3 = torch.cat([f_xl3,x4],1)
        f_x14 = self.xl4(f_xl3)
        f_xl4 = torch.cat([f_x14,x5],1)
        f_xl5 = self.xl5(f_xl4)

        b, c, h, w = x5.shape
        att = torch.flatten(f_xl5, 1).squeeze() # b 1 *7*7
        att = self.ln(att)
        # alpha = self.softmax(att)  # (batch_size,7*7)
        alpha = self.sigmoid(att)  # (batch_size,7*7)

        feature_1 = x5.reshape(b,h,w,c)
        # print(feature_1.shape, alpha.shape)

        SAM_weighted_encoding = feature_1 * alpha.reshape(b,h,w,1)  # (batch_size, encoder_dim)
        self.att_weight = alpha
        img = SAM_weighted_encoding.reshape(b,c,h,w)
        # img = self.conv(img)
        return img, alpha.reshape(b,7,7)


if __name__ == '__main__':
    vgg = Vgg16()
    print(vgg)
