import argparse
import os
import os.path as osp
import numpy as np
import PIL.Image
import skimage.io
import skimage.color as color
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


import sys
sys.path.append('/vis/home/arunirc/data1/Research/colorize-fcn/colorizer-fcn')
import utils
import data_loader


root = '/vis/home/arunirc/data1/datasets/ImageNet/images/'
out_path = '/data2/arunirc/Research/colorize-fcn/pytorch-fcn/tests/data_tests/'
GMM_PATH = '/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/gmm.pkl'
MEAN_L_PATH = '/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/mean_l.npy'
cuda = torch.cuda.is_available()



def main():
    dataset = data_loader.ColorizeImageNet(
                root, split='val', set='small',
                bins='soft', num_hc_bins=16,
                gmm_path=GMM_PATH, mean_l_path=MEAN_L_PATH)
    img, labels = dataset.__getitem__(0)
    gmm = dataset.gmm
    mean_l = dataset.mean_l

    img_file = dataset.files['val'][1]
    im_orig = skimage.io.imread(img_file)

    # ... predicted labels and input image (mean subtracted)
    labels = labels.numpy()
    img = img.squeeze().numpy()
    im_rgb = utils.colorize_image_hc(labels, img, gmm, mean_l)

    plt.imshow(im_rgb)
    plt.show()

    # 
    inputs = Variable(img)
    if cuda:
      inputs = inputs.cuda()
    outputs = model(inputs)
    # TODO: assertions
    # del inputs, outputs







if __name__ == '__main__':
    main()
