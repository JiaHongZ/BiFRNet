
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
from data_loader_OIC_FRM import getImg, imgLoader, Imgset, Imgset_train, imgLoader_train
from models.BIFRNet import BIFRNet
from models.VGG16_For_Comp import Vgg16_1 as Teacher

import utils.logger as log
import random
from torch.utils.data import DataLoader
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(100)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
def test(mynet):
    ############################
    # Test Loop
    ############################
    mynet.eval()
    zero_accuracy = 0
    for occ_level in occ_levels_test:

        if occ_level == 'ZERO':
            occ_types = ['']
        else:
            if dataset == 'pascal3d+':
                occ_types = ['', '_white', '_noise', '_texture']  # ['_white','_noise', '_texture', '']
            elif dataset == 'coco':
                occ_types = ['']

        for index, occ_type in enumerate(occ_types):
            # load images
            test_imgs, test_labels, masks = getImg('test', categories, dataset, data_path, categories_test, occ_level,
                                                   occ_type, bool_load_occ_mask=False)
            print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(
                len(test_imgs)))
            # get image loader
            test_imgset = Imgset(test_imgs, masks, test_labels, imgLoader, bool_square_images=False)

            test_loader = DataLoader(dataset=test_imgset, batch_size=batch_size, shuffle=True,drop_last=False)
            print('Testing')
            currect = 0
            count = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    img, _,_, lab = data
                    b, c, h, w = img.shape
                    img = img.cuda()
                    lab = lab.cuda()
                    with torch.no_grad():
                        pre,*_ = mynet(img)
                    _, lab_pre = torch.max(pre.data, 1)
                    currect += torch.sum(lab_pre == lab.data)
                    count += b
                current_rate = torch.true_divide(currect, count)
            # break
            print(occ_level, occ_type, current_rate)
            if occ_level == 'ZERO':
                zero_accuracy = current_rate
            mynet.train()
        return zero_accuracy
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # 只对未遮挡进行更新
        # sparse = torch.sum(x - x.mul(y)) / float(batch_size) # 不注意的地方
        distance = -torch.sum((x.mul(y) - y)) / float(batch_size)  # 注意的地方
        return distance
def train_mynet(train_loader):
    time_step = 3
    mynet = BIFRNet(time_step=time_step)
    teacher = Teacher()
    teacher = nn.DataParallel(teacher)
    teacher.load_state_dict(torch.load(r'model_zoo\BIFRNet\VGG16\coco\best.pth'),strict=True)
    teacher = teacher.module.vgg_pretrained_features.cuda()
    start = 0
    mynet = nn.DataParallel(mynet)
    mynet = mynet.cuda()

    mynet.load_state_dict(torch.load(path +'/best.pth'),strict=True)
    # mynet.module.att.parameters = teacher.cuda().parameters

    # 前面的feature部分不训练
    for para in teacher.parameters():
        para.requires_grad = False
    # for para in mynet.module.att.parameters():
    #     para.requires_grad = False
    print('-----start train model------')
    # test(model)
    train_loader = DataLoader(dataset=train_imgset, batch_size=batch_size, shuffle=True)

    optimiter = torch.optim.Adam(mynet.parameters(),lr=lr,betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0001)
    # optimiter1 = torch.optim.Adam(mynet.module.classifer.parameters(),lr=0.0001,betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0001)
    ######### Scheduler ###########
    warmup = True
    if warmup:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiter, 10, eta_min=1e-7)
    loss_f1 = torch.nn.CrossEntropyLoss().cuda()  # 与pattern的损失函数
    loss_f2 = My_loss().cuda()  # 与pattern的损失函数
    # loss_f2 = torch.nn.L1Loss().cuda()  # 与pattern的损失函数
    # loss_f2_2 = torch.nn.MultiLabelSoftMarginLoss().cuda()  # 与pattern的损失函数
    # loss_f2 = torch.nn.KLDivLoss().cuda()  # 与pattern的损失函数


    best = 0
    for epo in range(start+1,vgg_epoch):
        currect = 0
        count = 0
        for i, data in enumerate(train_loader):
            # print(i)
            img, img_org, occ, lab = data
            b, c, h, w = img.shape
            img = Variable(img).cuda()
            img_org = img_org.cuda()
            occ = occ.cuda()

            lab = Variable(lab).cuda()
            # with torch.no_grad():
            feature_occ, w, f_vgg = mynet.module.att(img)
            with torch.no_grad():
                feature_clear = teacher(img_org)
                # feature_clear,_ = teacher(img_org)
            pre, loss1 = mynet(None, feature_clear, feature_occ, lab, f_vgg)
            loss2 = loss_f1(pre,lab)
            loss3 = loss_f2(w, occ)
            # loss3_2 = loss_f2_2(w, occ)

            # print(w.shape, occ.shape)
            # loss3 = loss_f2(torch.log(torch.softmax(w,1)), torch.softmax(occ,1))

            loss = loss2+loss1.mean()+loss3*0.1


            optimiter.zero_grad()
            loss.backward()
            optimiter.step()

            _, lab_pre = torch.max(pre.data, 1)
            currect += torch.sum(lab_pre == lab.data)
            count += b

            # print('classifer',loss)
            # if i % 50 == 0:
            #     from PIL import Image
            #     # print(w[0])
            #     # print('occ',occ[0])
            #     img_recon = (w[0].cpu().detach().numpy().reshape(7, 7))
            #     # img_pil = Image.fromarray(img_recon)
            #     # img_recon = np.array(img_pil.resize((224, 224), Image.ANTIALIAS))
            #     plt.imshow(img_recon)
            #     plt.savefig('a.png')
            #     img_recon = (show(img[0].cpu()))
            #     plt.imshow(img_recon)
            #     plt.savefig('b.png')
            #     img_recon = (show(occ[0].cpu()))
            #     plt.imshow(img_recon)
            #     plt.savefig('c.png')
            #     print('asd', loss1.mean(), loss2)
        # print(mynet.module.knowledge.knowledge)
        logger.info('train correct rate:[{}] epoch:[{}] current learning rate:[{}]'.format((int(currect)/int(count)),epo,optimiter.param_groups[0]['lr']))

        if epo % 5 == 0 and epo > 0:
            scheduler.step()
            print('epoch', epo, ' current learning rate', optimiter.param_groups[0]['lr'])
        test_val_ = test(mynet)
        if test_val_ > best and test_val_ > 0.956:
            torch.save(mynet.state_dict(), path + '/'+str(epo)+'best.pth')
            best = test_val_

            print('now_best',best)

        logger.info('test correct rate:[{}] epoch:[{}] best:[{}]'.format(test_val_,epo, best))

if __name__ == '__main__':
    '''
        Init parameters
    '''
    batch_size = 64
    val_batch = 64
    batch_size_test = 64
    length = 'BIFRNet'

    name = 'coco'
    vgg_epoch = 200

    confidence = 1.5
    class_num = 12
    lr = 0.0001
    occ_levels_test = ['ZERO', 'ONE', 'FIVE', 'NINE']  # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]
    bool_mixture_model_bg = False  # True: use a mixture of background models per pixel, False: use one bg model for whole image
    bool_load_pretrained_model = False
    bool_train_with_occluders = False

    '''
        init model
    '''


    path = os.getcwd()
    path = os.path.join(path, 'model_zoo', 'BIFRNet', str(length),str(name))
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
    dataset = 'coco'
    data_path = r'F:\dateset\data_compnet\\'
    categories = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
       'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
    categories_test = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']

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

    # val_imgsets = []
    # if val_imgs:
    #     val_imgset = Imgset(val_imgs, val_masks, val_labels, imgLoader_train, bool_square_images=False)
    #     val_imgsets.append(val_imgset)

    train_loader = DataLoader(dataset=train_imgset, batch_size=batch_size, shuffle=True)
    # val_loaders = []
    #
    # for i in range(len(val_imgsets)):
    #     val_loader = DataLoader(dataset=val_imgsets[i], batch_size=val_batch, shuffle=True)
    #     val_loaders.append(val_loader)

    '''
        train
    '''
    # train(train_loader)
    train_mynet(train_loader)
