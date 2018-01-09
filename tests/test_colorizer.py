import argparse
import os
import os.path as osp
import numpy as np
import PIL.Image
import skimage.io
import skimage.color as color
from skimage import img_as_ubyte
import torch
from torch.autograd import Variable
import sys
sys.path.append('/vis/home/arunirc/data1/Research/colorize-fcn/colorizer-fcn')
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

# import models
# import utils
import data_loader

root = '/srv/data1/arunirc/datasets/ImageNet/images/'
cuda = torch.cuda.is_available()
GMM_PATH = '/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/gmm.pkl'
MEAN_L_PATH = '/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/mean_l.npy'


def test_color_gmm():
    print 'Entering: test_color'
    dataset = data_loader.ColorizeImageNet(
                root, split='val', set='tiny',
                bins='soft', num_hc_bins=16,
                gmm_path=GMM_PATH, mean_l_path=MEAN_L_PATH)

    img, labels = dataset.__getitem__(0)
    gmm = dataset.gmm

    labels = labels.numpy()
    img = img.squeeze().numpy()
    labels = labels.astype(gmm.means_.dtype)
    img = img.astype(gmm.means_.dtype)

    # expectation over GMM centroids
    hc_means = gmm.means_.astype(labels.dtype)
    im_hc = np.tensordot(labels, hc_means, (2,0)) 
    im_l = img + dataset.mean_l.astype(img.dtype)
    im_rgb = dataset.hue_chroma_to_rgb(im_hc, im_l)
    low, high = np.min(im_rgb), np.max(im_rgb)
    im_rgb = (im_rgb - low) / (high - low)
    im_out = img_as_ubyte(im_rgb)
    skimage.io.imsave("tests/output.png", im_out)

    img_file = dataset.files['val'][0]
    im_orig = skimage.io.imread(img_file)
    skimage.io.imsave("tests/orig.png", im_orig)



def main():
    test_color_gmm()





if __name__ == '__main__':
    main()
