#BIFRNet: A Brain-Inspired Feature Restoration DNN for Partially Occluded Image Recognition

Published in AAAI2023


## 1. Dataset
Download the dataset from [Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion](https://github.com/AdamKortylewski/CompositionalNets).


## 2. Test pertrained models 
- Modify the data_path in testing file: main_test_BIFRNet_pascal3d+.py (for pascal3d+), main_test_BIFRNet_coco.py (for coco) or others.
- Download the pretrained weights in:

链接：https://pan.baidu.com/s/1_hzmvV4JX0cRKdXobstMEA?pwd=ogd8 
提取码：ogd8 
--来自百度网盘超级会员V8的分享

- Put the downloaded folder 'BIFRNet' and 'VGG16' in 'model_zoo'.
  
  'model_zoo\BIFRNet\BIFRNet\best.pth'
  
  'model_zoo\BIFRNet\VGG16\best.pth''
    
- run main_test_BIFRNet_pascal3d+.py


## 3. Train your own model
Modify the data_path in training file: main_train_BIFRNet.py (for pascal3d+), main_train_BIFRNet_coco.py (for coco) or others.

Before you start to train BIFRNet, you should first training the VGG16 for the completion target in BIFRNet training.

run 
- main_vgg16_for_completion.py
  
Than, run
- main_test_BIFRNet_pascal3d+.py

 Some candidate model will be obtained in training stage. You can choose the best model.
 
 
 ## Reference
 
 @inproceedings{zhang2023bifrnet,
  title={BIFRNet: a brain-inspired feature restoration DNN for partially occluded image recognition},
  author={Zhang, Jiahong and Cao, Lihong and Lai, Qiuxia and Li, Bingyao and Qin, Yunxiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={12},
  pages={15296--15304},
  year={2023}
}
