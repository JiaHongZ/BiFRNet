import torch
from torch import nn
from torchvision import models
import numpy as np
from models.core import GenerationCoreK as GenerationCore
from torch.distributions import Normal


class Vgg16(torch.nn.Module):
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

    def __init__(self, encoder_dim, classes):
        super().__init__()
        # self.conv = nn.Conv2d(512*2, 512, 3, 1, 1)
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
        # x = self.conv(x)
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
def getmap(x,y,patch_size):
    x = int(x)
    y = int(y)
    position_map = np.zeros((patch_size,patch_size))
    for i in range(patch_size):
        for j in range(patch_size):
            d = np.sqrt((i-x)**2 + (j-y)**2)
            # if 1 * np.exp(-d/14) > 0.9:
            #     position_map[i][j] = 1
            position_map[i][j] = 1 * np.exp(-d / 6)
    position_map = position_map - 0.4
    return position_map
class SAM(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, input_channel, hidden, image_size, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(SAM, self).__init__()
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
        self.sigmoid = nn.Sigmoid()

        self.att_weight = None # 记录
        self.map = torch.tensor(getmap(3,3,7).astype(np.float32)).cuda()

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
        alpha = self.sigmoid(att)  # (batch_size,7*7)

        feature_1 = x5.reshape(b,h,w,c)
        alpha = alpha.reshape(b,h,w)


        attention_weighted_encoding = feature_1 * alpha.reshape(b,h,w,1)  # (batch_size, encoder_dim)
        self.att_weight = alpha
        img = attention_weighted_encoding.reshape(b,c,h,w)
        return img, alpha.reshape(b,1,7,7), x5
class Know(torch.nn.Module):
    def __init__(self, encoder_dim, classes):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(encoder_dim, classes, 1, 1, 0),
            nn.LayerNorm([classes, 7, 7]),
        )
        self.classes = classes
        self.knowledge = nn.Parameter(
            torch.Tensor(torch.rand(classes, 7, 7)), requires_grad=True)
    def forward(self, x, lab):
        # x b 512 7 7
        b,c,h,w = x.shape
        x = self.encode(x) # b 12 7 7
        # print('lab',lab)
        lab = nn.functional.one_hot(lab,num_classes=self.classes).unsqueeze(2).unsqueeze(3)  # b 12 1
        k_ = x * lab
        k_ = k_.sum(0)/b
        # 不然的话会出现0.导致求导出问题
        for i in range(len(k_)):
            if k_[i].sum() == 0:
                k_[i] = self.knowledge[i]
        # kl divergence
        d = torch.nn.functional.kl_div(torch.softmax(k_,1), torch.softmax(self.knowledge,1))
        return d
class FRM(nn.Module):
    def __init__(self, L=3, sigma=0.5, shared_core=False):
        super(FRM, self).__init__()

        # Number of generative layers
        self.L = L
        self.sigma = sigma

        # Generation network
        self.shared_core = shared_core
        if shared_core:
            self.generation_core = GenerationCore()
        else:
            self.generation_core = nn.ModuleList([GenerationCore() for _ in range(L)])

        self.eta_g = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, f_visual, k, f_visual_encod, f_vgg):
        '''
            f_visual b 512 7 7
            k b 12 7 7
            f_visual_encod b 12 7 7
        '''
        B, M, *_ = f_visual.size()

        # Generator initial state
        c_g = k.squeeze(0).repeat(B,1,1,1)
        h_g = f_visual_encod
        u = f_vgg

        elbo = 0
        for l in range(self.L):
            if self.shared_core:
                c_g, h_g, u = self.generation_core(f_visual, c_g, h_g, u)
            else:
                c_g, h_g, u = self.generation_core[l](f_visual, c_g, h_g, u)
        u = self.eta_g(u)
        elbo += torch.sum(Normal(u, self.sigma).log_prob(f_visual), dim=[1, 2, 3])
        return elbo, u
class BIFRNet(nn.Module):
    def __init__(self, in_channels=3, hidden=512, image_size=7, z_dim=512, decoder_dim=1024, attention_dim=2048, encoder_dim=1024, classes=12, time_step=3):
        super().__init__()
        self.classes = classes
        self.z_dim = z_dim
        self.image_size = image_size
        self.time_step = time_step
        self.decoder_dim = decoder_dim
        self.visual_dim = hidden*image_size*image_size
        self.conception_dim = classes*7*7
        # ventral visual pathway and dorsal visual pathway
        self.att = SAM(in_channels, hidden, image_size,  self.z_dim, attention_dim)
        # knowledge
        self.knowledge = Know(hidden,classes)
        # completion
        self.frm = FRM(L=7)
        self.classifer = Vgg16_classifier(hidden*7*7,classes)
    def forward(self, x, f_clear=None, f_visual=None, lab=None, f_vgg=None):
        if lab == None: # 测试情况
            b,c,h,w = x.size()
            f_visual, visual_w, f_vgg = self.att(x)

            k = self.knowledge.knowledge
            f_visual_encod = self.knowledge.encode(f_visual) # b 512 1 1
            _, f_all = self.frm(f_visual, k, f_visual_encod, f_vgg)
            result1 = self.classifer(f_all)
            return result1, visual_w, f_vgg, f_all, f_visual, f_visual_encod
        else:
            k_energy = self.knowledge(f_clear,lab)
            k = self.knowledge.knowledge
            f_visual_encod = self.knowledge.encode(f_visual) # b 512 1 1
            elbo, f_all = self.frm(f_visual, k, f_visual_encod, f_vgg)
            l2loss = nn.functional.mse_loss(f_all, f_clear)
            result1 = self.classifer(f_all)
            return result1, l2loss-k_energy

class Vgg16_1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
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
        y = torch.flatten(x, 1)
        y = self.classifier(y)
        return y
if __name__ == '__main__':
    vgg = Vgg16()
    print(vgg)
