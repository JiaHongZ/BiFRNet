
import os

import torch
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

show = ToPILImage()
import numpy as np
import utils.logger as log
from models.BIFRNet import BIFRNet, FRM
import torch.nn as nn
from data_loader import getImg, imgLoader, Imgset
from torch.utils.data import DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(100)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        sparse = torch.sum(x - x.mul(y)) / float(batch_size) # 除的越小越稀疏
        distance = -torch.sum((x.mul(y) - y)) / float(batch_size)
        return distance,sparse

def test(model,FRM,test_data, batch_size):
    model.eval()
    model.load_state_dict(torch.load(os.path.join(path,'best.pth')),strict=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    print('Testing')
    currect = 0
    count = 0
    for i, data in enumerate(test_loader):
        img, _, lab = data
        b, c, h, w = img.shape
        img = Variable(img).cuda()
        lab = Variable(lab).cuda()
        with torch.no_grad():
            pre,*_ = mynet(img)
        _, lab_pre = torch.max(pre.data, 1)
        currect += torch.sum(lab_pre == lab.data)
        count += b
    current_rate = torch.true_divide(currect, count)
    return current_rate

if __name__ == '__main__':
    '''
        Init parameters
    '''
    batch_size = 64
    batch_size_test = 64
    length = 'BIFRNet'

    name = 'pascal3d+'
    # length = 'OccNet_K_rebuttal'
    # name = 'OccNet_K_dir_all1'
    # name = 'pascal3d+'

    vgg_epoch = 200

    recall_steps = 4
    confidence = 2
    lr = 0.0001
    classes = 12
    ###################
    # Test parameters #
    ###################
    occ_levels = ['ZERO', 'ONE', 'FIVE', 'NINE']  # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]
    # occ_levels = ['FIVE']  # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]
    bool_load_pretrained_model = False  # False if you want to load initialization (see Initialization_Code/)
    bool_mixture_model_bg = False  # use maximal mixture model or sum of all mixture models, not so important
    bool_multi_stage_model = False  # this is an old setup
    # dataset = 'pascal3d+'
    dataset = 'pascal3d+'
    bool_train_with_occluders = False

    categories_train = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                        'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
    data_path = r'F:\dateset\data_compnet\\'

    # categories = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
    #               'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
    categories = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']

    '''
        init model
    '''
    mynet = BIFRNet()
    mynet = nn.DataParallel(mynet)
    mynet = mynet.cuda()
    #

    path = os.getcwd()
    path = os.path.join(path, 'model_zoo', 'BIFRNet', str(length), str(name))
    # model path
    if not os.path.exists(path):
        os.makedirs(path)
        print('path create')

    # log path
    log_path = os.path.join(path, 'v15test_log_' + str(length) + '_conf' + str(confidence) + '.log')
    logger = log.get_logger(log_path)
    ############################
    # Test Loop
    ############################
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
            # compute test accuracy
            logger.info('dataset:[{}] Occ_level:[{}], Occ_type:[{}] confidence:[{}] recall_steps:[{}]'.format(dataset,
                                                                                                              occ_level,
                                                                                                              occ_type,
                                                                                                              confidence,
                                                                                                              recall_steps))

            acc = test(mynet,FRM, test_data=test_imgset, batch_size=batch_size_test)
            logger.info('Occ test result: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc))

            out_str = 'Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
            print(out_str)
    '''
        train
    '''
    # train(train_loader)
    # train_mynet(train_loader)
