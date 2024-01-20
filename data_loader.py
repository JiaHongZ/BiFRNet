import os
import torch
import cv2
import glob
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

def pad_to_size(x, to_size):
	padding = [(to_size[1] - x.shape[3]) // 2, (to_size[1] - x.shape[3]) - (to_size[1] - x.shape[3]) // 2, (to_size[0] - x.shape[2]) // 2, (to_size[0] - x.shape[2]) - (to_size[0] - x.shape[2]) // 2]
	return F.pad(x, padding)

def myresize(img, dim, tp):
	H, W = img.shape[0:2]
	if tp == 'short':
		if H <= W:
			ratio = dim / float(H)
		else:
			ratio = dim / float(W)

	elif tp == 'long':
		if H <= W:
			ratio = dim / float(W)
		else:
			ratio = dim / float(H)

	return cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

def getImg(mode,categories, dataset, data_path, cat_test=None, occ_level='ZERO', occ_type=None, bool_load_occ_mask = False):

	if mode == 'train':
		train_imgs = []
		train_labels = []
		train_masks = []
		for category in categories:
			if dataset == 'pascal3d+':
				if occ_level == 'ZERO':
					filelist = data_path + 'pascal3d+_occ/' + category + '_imagenet_train' + '.txt'
					img_dir = data_path + 'pascal3d+_occ/TRAINING_DATA/' + category + '_imagenet'
			elif dataset == 'coco':
				if occ_level == 'ZERO':
					img_dir = data_path +'coco_occ/{}_zero'.format(category)
					filelist = data_path +'coco_occ/{}_{}_train.txt'.format(category, occ_level)

			with open(filelist, 'r') as fh:
				contents = fh.readlines()
			fh.close()
			img_list = [cc.strip() for cc in contents]
			label = categories.index(category)
			for img_path in img_list:
				if dataset=='coco':
					if occ_level == 'ZERO':
						img = img_dir + '/' + img_path + '.jpg'
					else:
						img = img_dir + '/' + img_path + '.JPEG'
				else:
					img = img_dir + '/' + img_path + '.JPEG'
				occ_img1 = []
				occ_img2 = []
				train_imgs.append(img)
				train_labels.append(label)
				train_masks.append([occ_img1,occ_img2])

		return train_imgs, train_labels, train_masks

	else:
		test_imgs = []
		test_labels = []
		occ_imgs = []
		for category in cat_test:
			if dataset == 'pascal3d+':
				filelist = data_path + 'pascal3d+_occ/' + category + '_imagenet_occ.txt'
				img_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level
				if bool_load_occ_mask:
					if  occ_type=='':
						occ_mask_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask_object'
					else:
						occ_mask_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask'
					occ_mask_dir_obj = data_path + 'pascal3d+_occ/0_old_masks/'+category+'_imagenet_occludee_mask/'
			elif dataset == 'coco':
				if occ_level == 'ZERO':
					img_dir = data_path+'coco_occ/{}_zero'.format(category)
					filelist = data_path+'coco_occ/{}_{}_test.txt'.format(category, occ_level)
				else:
					img_dir = data_path+'coco_occ/{}_occ'.format(category)
					filelist = data_path+'coco_occ/{}_{}.txt'.format(category, occ_level)

			if os.path.exists(filelist):
				with open(filelist, 'r') as fh:
					contents = fh.readlines()
				fh.close()
				img_list = [cc.strip() for cc in contents]
				label = categories.index(category)
				for img_path in img_list:
					if dataset != 'coco':
						if occ_level=='ZERO':
							img = img_dir + occ_type + '/' + img_path[:-2] + '.JPEG'
							occ_img1 = []
							occ_img2 = []
						else:
							img = img_dir + occ_type + '/' + img_path + '.JPEG'
							if bool_load_occ_mask:
								occ_img1 = occ_mask_dir + '/' + img_path + '.JPEG'
								occ_img2 = occ_mask_dir_obj + '/' + img_path + '.png'
							else:
								occ_img1 = []
								occ_img2 = []

					else:
						img = img_dir + occ_type + '/' + img_path + '.jpg'
						occ_img1 = []
						occ_img2 = []

					test_imgs.append(img)
					test_labels.append(label)
					occ_imgs.append([occ_img1,occ_img2])
			else:
				print('FILELIST NOT FOUND: {}'.format(filelist))
		return test_imgs, test_labels, occ_imgs

def imgLoader(img_path,mask_path,bool_resize_images=True,bool_square_images=False):

	input_image = Image.open(img_path)
	if bool_resize_images:
		if bool_square_images:
			input_image.resize((224,224),Image.ANTIALIAS)
		else:
			sz=input_image.size
			min_size = np.min(sz)
			if min_size!=224:
				input_image = input_image.resize((np.asarray(sz) * (224 / min_size)).astype(int),Image.ANTIALIAS)
	preprocess = transforms.Compose([
		transforms.Resize((224,224),Image.ANTIALIAS),
		transforms.ToTensor(),
		# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	img = preprocess(input_image)

	if mask_path[0]:
		mask1 = cv2.imread(mask_path[0])[:, :, 0]
		# mask1 = myresize(mask1, 224, 'short')
		mask1 = cv2.resize(mask1, (224, 224))
		try:
			mask2 = cv2.imread(mask_path[1])[:, :, 0]
			mask2 = mask2[:mask1.shape[0], :mask1.shape[1]]
		except:
			mask = mask1
		try:
			mask = ((mask1 == 255) * (mask2 == 255)).astype(np.float)
		except:
			mask = mask1
	else:
		mask = np.ones((img.shape[0], img.shape[1])) * 255.0

	mask = torch.from_numpy(mask)
	return img,mask

class Imgset():
	def __init__(self, imgs, masks, labels, loader,bool_square_images=False):
		self.images = imgs
		self.masks 	= masks
		self.labels = labels
		self.loader = loader
		self.bool_square_images = bool_square_images

	def __getitem__(self, index):
		fn = self.images[index]
		label = self.labels[index]
		mask = self.masks[index]
		img,mask = self.loader(fn,mask,bool_resize_images=True,bool_square_images=self.bool_square_images)
		return img, mask, label

	def __len__(self):
		return len(self.images)
