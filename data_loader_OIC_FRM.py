import os
import torch
import cv2
import glob
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import PIL
import pickle

with open('utils/imgbag.txt','rb') as file:
	dictionary = pickle.load(file)

def getretinamap(x,y,patch_size):
	x = int(x)
	y = int(y)
	position_map = np.zeros((patch_size,patch_size))
	for i in range(patch_size):
		for j in range(patch_size):
			d = np.sqrt((i-x)**2 + (j-y)**2)
			position_map[i][j] = 1 * np.exp(-d/10)
	return position_map

retinamap = getretinamap(3,3,7)

def create_occ(occ_ratio,img):
	img = np.array(img)
	patch_row,patch_col, c = img.shape
	x = np.random.randint(0,patch_row)
	y = np.random.randint(0,patch_col)
	occ = np.ones((patch_row,patch_col))
	# print('x,y',x, y)
	rand_float = np.random.rand()
	rand_float2 = np.random.rand()
	if c!=3:
		rand_float = 0
	isocc = True
	if rand_float > 0.2: # 0.8的概率遮挡
		xx = int(0 if (x-occ_ratio/2) <0 else (x-occ_ratio/2))
		xy = int(patch_row if (x+occ_ratio/2) >=patch_row else (x+occ_ratio/2))
		yx = int(0 if (y-occ_ratio/2) <0 else (y-occ_ratio/2))
		yy = int(patch_col if (y+occ_ratio/2) >=patch_col else (y+occ_ratio/2))
		# 防止出现全是遮挡物
		if xx==0 and yx==0 and xy>=(patch_row-50) and yy>=(patch_col - 50):
			xy = patch_row - 50
			yy = patch_col - 50
		if rand_float2 < 0.3:
			img[xx:xy, yx:yy, :] = np.random.rand(xy-xx, yy-yx, 3) * 255
		elif rand_float2 >= 0.3 and rand_float2 < 0.6:
			img[xx:xy, yx:yy, :] = 255
		else:
			# img[xx:xy, yx:yy, :] = 0 # 原来是有object遮挡
			slice = np.random.choice(len(dictionary), 1)
			oc = dictionary[int(slice)].resize((int(yy-yx), int(xy-xx)), Image.ANTIALIAS)
			oc = np.array(oc)
			if len(oc.shape) == 3:
				img[xx:xy, yx:yy, :] = oc[:,:,:3]
			# import matplotlib.pyplot as plt
			# plt.imshow(img)
			# plt.show()
		occ[xx:xy, yx:yy] = 0
		isocc = True

	else:
		isocc = False

		# occ[0:30, 0:30] = 0.5
		# occ[0:30, -30:] = 0.5
		# occ[-30:, 0:30] = 0.5
		# occ[-30:, -30:] = 0.5
	occ = Image.fromarray(occ)
	# occ = occ.resize((14,14))
	return Image.fromarray(img), occ, isocc

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
	# preprocess = transforms.Compose([transforms.Resize((224,224),Image.ANTIALIAS),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	preprocess = transforms.Compose([
		transforms.Resize((224,224),Image.ANTIALIAS),
		transforms.ToTensor(),
		# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	img = preprocess(input_image)
	# print('maskpath',mask_path)
	if mask_path[0]:
		mask1 = cv2.imread(mask_path[0])[:, :, 0]
		mask1 = myresize(mask1, 224, 'short')
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
	mask = Image.fromarray(mask)
	preprocess = transforms.Compose([transforms.Resize((224,224),Image.ANTIALIAS),transforms.ToTensor()])
	mask = preprocess(mask)
	# mask = torch.from_numpy(mask)
	return img,0,mask

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
		img,img_org,occ = self.loader(fn,mask,bool_resize_images=True,bool_square_images=self.bool_square_images)
		return img,img_org, occ, label

	def __len__(self):
		return len(self.images)

def imgLoader_train(img_path,mask_path,bool_resize_images=True,bool_square_images=False):

	input_image = Image.open(img_path)
	if bool_resize_images:
		if bool_square_images:
			input_image.resize((224,224),Image.ANTIALIAS)
		else:
			sz=input_image.size
			min_size = np.min(sz)
			if min_size!=224:
				input_image = input_image.resize((np.asarray(sz) * (224 / min_size)).astype(int),Image.ANTIALIAS)

	# preprocess = transforms.Compose([transforms.Resize((224,224),Image.ANTIALIAS),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	preprocess = transforms.Compose([
									transforms.Resize((224,224),Image.ANTIALIAS),
            						 # transforms.RandomHorizontalFlip(),
                                     # transforms.RandomVerticalFlip(),
                                     # transforms.RandomRotation(degrees=(10, 80)),
									 transforms.ToTensor(),
									 # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
									 ])
	preprocess2 = transforms.Compose([
									transforms.Resize((7, 7), Image.ANTIALIAS),
									 # transforms.RandomHorizontalFlip(),
									 # transforms.RandomVerticalFlip(),
									 # transforms.RandomRotation(degrees=(10, 80)),
									 transforms.ToTensor(),
								     # transforms.Normalize(mean=[0], std=[1]),
									  ])
	# variabel occ
	occl = np.random.randint(50,400)
	# occl = np.random.randint(100,450)
	img_org = input_image.copy()
	input_image, occ, isocc = create_occ(occl,input_image)

	# import matplotlib.pyplot as plt
	# plt.imshow(input_image)
	# plt.show()
	# import matplotlib.pyplot as plt
	# plt.imshow(img_org)
	# plt.show()
	# import matplotlib.pyplot as plt
	# plt.imshow(np.array(input_image)*np.array(occ)[:,:,np.newaxis].repeat(3,axis=2))
	# plt.show()
	# import matplotlib.pyplot as plt
	# plt.imshow(occ)
	# plt.show()

	img_occ = preprocess(input_image)
	img = preprocess(img_org)
	# occ[occ<0.8] = 0
	# occ[occ>=0.8] = 1
	if isocc:
		occ = preprocess2(occ)
		occ[occ<0.9] = 0
		occ[occ>=0.9] = 1
	else:
		occ = torch.Tensor(np.array(retinamap)).reshape(1,7,7)
		# occ = torch.Tensor(np.ones((1, 7, 7)))
		# occ[:, 0, 0] = 0.5
		# occ[:, 0, 6] = 0.5
		# occ[:, 6:, 6] = 0.5
		# occ[:, 6:, 0] = 0.5
		# print(occ)
	# print(occ)

	if mask_path[0]:
		mask1 = cv2.imread(mask_path[0])[:, :, 0]
		mask1 = myresize(mask1, 224, 'short')
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
	# occ = torch.Tensor(np.array(occ))
	return img_occ,img,occ
class Imgset_train(): #加入图像增强
	def __init__(self, imgs, masks, labels, loader,bool_square_images=False):
		self.images = imgs
		self.masks 	= masks
		self.labels = labels
		self.loader = imgLoader_train
		self.bool_square_images = bool_square_images

	def __getitem__(self, index):
		fn = self.images[index]
		label = self.labels[index]
		mask = self.masks[index]
		img,img_org,occ = self.loader(fn,mask,bool_resize_images=True,bool_square_images=self.bool_square_images)
		return img,img_org,occ, label

	def __len__(self):
		return len(self.images)
def save_checkpoint(state, filename, is_best):
	if is_best:
		print("=> Saving new checkpoint")
		torch.save(state, filename)
	else:
		print("=> Validation Accuracy did not improve")
