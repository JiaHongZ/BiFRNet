import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
show = ToPILImage()
import torch
import numpy as np
# 测试
def show_img(img, lab):
    # 36 可视化时6行6列
    img = img.cpu()
    fig,ax = plt.subplots(6,6,figsize=(20,20))
    for i in range(6):
        for j in range(6):
            ax[i][j].imshow(show(img[i*6+j]))
            # ax[i][j].axes.get_yaxis.set_visible(False)
    plt.savefig('a.png')
    plt.show()

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def create_occ(patch_row,occ_ratio):
    '''
        input: patch_row is the h_num or w_num of the image patches
        occ_ratio
        output: occ list where the occ patch is 0, 0 shape[(occ_ratio-1)*2+1]

        the expected occ list need occlusion area in neigborhood
    '''
    x = np.random.randint(0,patch_row)
    y = np.random.randint(0,patch_row)
    occ = np.ones((patch_row,patch_row))
    # print('x,y',x, y)
    for i in range(patch_row):
        for j in range(patch_row):
            if abs(i-x) < occ_ratio and abs(j-y)<occ_ratio:
                # print('a',x, y)
                occ[i,j] = 0
    return occ.flatten()

if __name__ == '__main__':
    # x, y = create_occ(14,4)
    # print(x)
    # print(y)
    for i in range(100):
        print(np.random.randint(0,4))