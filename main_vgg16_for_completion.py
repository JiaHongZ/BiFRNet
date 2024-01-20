
from collections import namedtuple
import torch
from torch.autograd import Variable
import os
from torchvision.transforms import ToPILImage
import torchvision.models as models
show = ToPILImage()
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import torch.nn as nn
from data_loader_OIC import getImg, imgLoader, Imgset, Imgset_train, imgLoader_train
from models.VGG16_For_Comp import Vgg16_1 as OccNet
import utils.logger as log
import random
from torch.utils.data import DataLoader
from sklearn import manifold, datasets
import utils.image_utils as img_utils
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(100)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def test(model):
    ############################
    # Test Loop
    ############################
    model.eval()
    for occ_level in occ_levels:

        if occ_level == 'ZERO':
            occ_types = ['']
        else:
            if dataset == 'pascal3d+':
                occ_types = ['', '_white', '_noise', '_texture']  # ['_white','_noise', '_texture', '']
            elif dataset == 'coco':
                occ_types = ['']

        for index, occ_type in enumerate(occ_types):
            # load images
            test_imgs, test_labels, masks = getImg('test', categories_train, dataset, data_path, categories, occ_level,
                                                   occ_type, bool_load_occ_mask=False)
            print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(
                len(test_imgs)))
            # get image loader
            test_imgset = Imgset(test_imgs, masks, test_labels, imgLoader, bool_square_images=False)

            test_loader = DataLoader(dataset=test_imgset, batch_size=batch_size, shuffle=True,drop_last=True)
            print('Testing')
            currect = 0
            count = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    img, _, lab = data
                    b, c, h, w = img.shape
                    img = img.cuda()
                    lab = lab.cuda()
                    with torch.no_grad():
                        pre = model(img)
                    _, lab_pre = torch.max(pre.data, 1)
                    currect += torch.sum(lab_pre == lab.data)
                    count += b
                current_rate = torch.true_divide(currect, count)
            # break
            model.train()
            return current_rate

def train_mynet(train_loader):
    # mynet = MyNet(patch_size=16,img_size=224, in_channels=3,hidden_channels=8, n_blocks=1, n_classes=12)
    # mynet = mynet.cuda()
    # mynet = models.vgg16(pretrained=False).cuda()
    time_step = 3
    mynet = OccNet()
    # mynet = MyViT('B_16_imagenet1k',num_layers=6,image_size=224,num_classes=12,dropout_rate=0, pretrained=True)
    start = 0
    mynet = nn.DataParallel(mynet)

    # mynet.load_state_dict(torch.load(path +'/best.pth'),strict=True)
    mynet = mynet.cuda()

    print('-----start train model------')
    # test(model)
    train_loader = DataLoader(dataset=train_imgset, batch_size=batch_size, shuffle=True)
    optimiter = torch.optim.Adam(mynet.parameters(),lr=0.0001,betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0001)
    ######### Scheduler ###########
    warmup = True
    if warmup:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiter, 10, eta_min=1e-7)
    loss_f1 = torch.nn.CrossEntropyLoss().cuda()  # 与pattern的损失函数
    # loss_f2 = torch.nn.MSELoss().cuda()
    loss_f2 = torch.nn.MultiLabelSoftMarginLoss()
    best = 0
    for epo in range(start+1,vgg_epoch):
        currect = 0
        count = 0
        # for k, v in mynet.named_parameters():
        #     if 'block' in k:
        #         v.requires_grad = False
        for i, data in enumerate(train_loader):
            # print(i)
            img, occ, lab = data
            b, c, h, w = img.shape
            # img, occ = create_occ(b, 50, img)
            img_show = ToPILImage()(img.cpu()[0])
            # img = img.detach().cpu().numpy().reshape(224,224,3)
            # plt.imshow(img_show)
            # plt.show()
            # break
            # print(occ[0][0])
            # plt.imshow(occ.cpu()[0][0])
            # plt.show()
            img = Variable(img).cuda()
            occ = occ.cuda()
            lab = Variable(lab).cuda()
            pre = mynet(img)
            # print(visual_w,occ)

            # print(visual_w[-1],occ)
            # loss1 = loss_f1(concep_w,lab)
            loss2 = loss_f1(pre, lab)
            # loss3 = loss_f2(visual_w,occ)
            # print('loss1',loss1,'loss2',loss2,'loss3',loss3)
            # print(loss1,loss,loss2)
            loss = loss2
            optimiter.zero_grad()
            loss.backward()
            optimiter.step()
            _, lab_pre = torch.max(pre.data, 1)

            currect += torch.sum(lab_pre == lab.data)
            count += b

        logger.info('train correct rate:[{}] epoch:[{}] current learning rate:[{}]'.format((int(currect)/int(count)),epo,optimiter.param_groups[0]['lr']))

        if epo % 5 == 0 and epo > 0:
            scheduler.step()
            print('epoch', epo, ' current learning rate', optimiter.param_groups[0]['lr'])
        test_val_ = test(mynet)
        if test_val_ > best:
            torch.save(mynet.state_dict(), path +'/best.pth')
            best = test_val_
        torch.save(mynet.state_dict(), path + '/now.pth')
        logger.info('test correct rate:[{}] epoch:[{}]'.format(test_val_,epo))

if __name__ == '__main__':
    '''
        Init parameters
    '''
    batch_size = 64
    val_batch = 64
    batch_size_test = 64
    length = 'VGG16'
    name = 'pascal3d+'
    vgg_epoch = 200


    confidence = 1.5
    class_num = 12
    lr = 0.0001
    occ_levels = ['ZERO', 'ONE', 'FIVE', 'NINE']  # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]

    bool_mixture_model_bg = False  # True: use a mixture of background models per pixel, False: use one bg model for whole image
    bool_load_pretrained_model = False
    bool_train_with_occluders = False

    '''
        init model
    '''



    #

    path = os.getcwd()
    path = os.path.join(path, 'model_zoo', 'BIFRNet', str(length), str(name))
    # path = os.path.join(path, 'model_zoo', 'mynetViT_pascal3d+_occ', str(length))
    # model path
    if not os.path.exists(path):
        os.makedirs(path)
        print('path create')

    # log path
    log_path = os.path.join(path, 'train_log_'+str(length)+'_conf'+str(confidence)+'.log')
    print(log_path)
    logger = log.get_logger(log_path)
    '''
        train dataset
    '''
    train_imgs = []
    train_masks = []
    train_labels = []
    val_imgs = []
    val_labels = []
    val_masks = []
    categories_train = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
       'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
    dataset = 'pascal3d+'
    data_path = r'F:\dateset\data_compnet\\'
    categories = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
       'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

    if bool_train_with_occluders:
        occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
    else:
        occ_levels_train = ['ZERO']
    # get training and validation images
    for occ_level in occ_levels_train:
        if occ_level == 'ZERO':
            occ_types = ['']
            train_fac = 0.9
        else:
            occ_types = ['_white', '_noise', '_texture', '']
            train_fac = 0.1

        for occ_type in occ_types:
            imgs, labels, masks = getImg('train', categories_train, dataset, data_path, categories, occ_level, occ_type,
                                         bool_load_occ_mask=False)
            nimgs = len(imgs)
            for i in range(nimgs):
                if (random.randint(0, nimgs - 1) / nimgs) <= train_fac:
                    train_imgs.append(imgs[i])
                    train_labels.append(labels[i])
                    train_masks.append(masks[i])
                elif not bool_train_with_occluders:
                    val_imgs.append(imgs[i])
                    val_labels.append(labels[i])
                    val_masks.append(masks[i])

    print('Total imgs for train ' + str(len(train_imgs)))
    print('Total imgs for val ' + str(len(val_imgs)))
    train_imgset = Imgset(train_imgs, train_masks, train_labels, imgLoader_train, bool_square_images=False)

    val_imgsets = []
    if val_imgs:
        val_imgset = Imgset(val_imgs, val_masks, val_labels, imgLoader_train, bool_square_images=False)
        val_imgsets.append(val_imgset)

    train_loader = DataLoader(dataset=train_imgset, batch_size=batch_size, shuffle=True)
    val_loaders = []

    for i in range(len(val_imgsets)):
        val_loader = DataLoader(dataset=val_imgsets[i], batch_size=val_batch, shuffle=True)
        val_loaders.append(val_loader)

    '''
        train
    '''
    # train(train_loader)
    train_mynet(train_loader)
