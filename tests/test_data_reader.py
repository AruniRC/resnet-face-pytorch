import argparse
import os
import os.path as osp

import numpy as np
import PIL.Image
import skimage.io
import skimage.color as color
import torch
from torch.autograd import Variable

import sys
sys.path.append('/vis/home/arunirc/data1/Research/colorize-fcn/colorizer-fcn')
import models
import train
import utils
import data_loader

root = '/vis/home/arunirc/data1/datasets/ImageNet/images/'


def test_single_read():
    print 'Entering: test_single_read'
    dataset = data_loader.ColorizeImageNet(root, split='train', set='small')
    img, lbl = dataset.__getitem__(0)
    assert len(lbl)==2 
    assert np.min(lbl[0].numpy())==0
    assert np.max(lbl[0].numpy())==30
    print 'Test passed: test_single_read'


def test_single_read_dimcheck():
    print 'Entering: test_single_read_dimcheck'
    dataset = data_loader.ColorizeImageNet(root, split='train', set='small')
    img, lbl = dataset.__getitem__(0)
    assert len(lbl)==2
    im_hue = lbl[0].numpy()
    im_chroma = lbl[1].numpy()
    assert im_chroma.shape==im_hue.shape, \
            'Labels (Hue and Chroma maps) should have same dimensions.'
    print 'Test passed: test_single_read_dimcheck'


def test_train_loader():
    print 'Entering: test_train_loader'
    train_loader = torch.utils.data.DataLoader(
        data_loader.ColorizeImageNet(root, split='train', set='small'),
        batch_size=1, shuffle=False)
    dataiter = iter(train_loader)
    img, label = dataiter.next()
    assert len(label)==2, \
        'Network should predict a 2-tuple: hue-map and chroma-map.'
    im_hue = label[0].numpy()
    im_chroma = label[1].numpy()
    assert im_chroma.shape==im_hue.shape, \
            'Labels (Hue and Chroma maps) should have same dimensions.'
    print 'Test passed: test_train_loader'


def test_dataset_read():
    '''
        Read through the entire dataset.
    '''
    dataset = data_loader.ColorizeImageNet(\
                root, split='train', set='small')

    for i in xrange(len(dataset)):
        # if i > 44890: # HACK: skipping over some stuff
        img_file = dataset.files['train'][i]
        img, lbl = dataset.__getitem__(i)
        assert type(lbl) == torch.FloatTensor
        assert type(img) == torch.FloatTensor
        print 'iter: %d,\t file: %s,\t imsize: %s' % (i, img_file, img.size())


def test_cmyk_read():
    '''
        Handle CMYK images -- skip to previous image.
    '''
    print 'Entering: test_cmyk_read'
    dataset = data_loader.ColorizeImageNet(\
                root, split='train', set='small')
    idx = 44896
    img_file = dataset.files['train'][idx]
    im1 = PIL.Image.open(img_file)
    im1 = np.asarray(im1, dtype=np.uint8)
    assert im1.shape[2]==4, 'Check that selected image is indeed CMYK.'
    img, lbl = dataset.__getitem__(idx)    
    print 'Test passed: test_cmyk_read'


def test_grayscale_read():
    '''
        Handle single-channel images -- skip to previous image.
    '''
    print 'Entering: test_grayscale_read'
    dataset = data_loader.ColorizeImageNet(root, split='train', set='small')
    idx = 4606
    img_file = dataset.files['train'][idx]
    im1 = PIL.Image.open(img_file)
    im1 = np.asarray(im1, dtype=np.uint8)
    assert len(im1.shape)==2, 'Check that selected image is indeed grayscale.'
    img, lbl = dataset.__getitem__(idx)    
    print 'Test passed: test_grayscale_read'


def test_rgb_hsv():
    # DEFER
    dataset = data_loader.ColorizeImageNet(\
                root, split='train', set='small')
    img_file = dataset.files['train'][100]
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=np.uint8)
    assert np.max(img.shape) == 400


def test_soft_bins():
    dataset = \
        data_loader.ColorizeImageNet(root, split='train', set='small', 
                                     bins='soft')
    img, lbl = dataset.__getitem__(0)    
    assert type(lbl) == torch.FloatTensor
    assert type(img) == torch.FloatTensor
    print 'Test passed: test_soft_bins'


def test_lowpass_image():
    dataset = \
        data_loader.ColorizeImageNet(root, split='train', set='small', 
                                     bins='soft', img_lowpass=8)
    img, lbl = dataset.__getitem__(0)    
    assert type(lbl) == torch.FloatTensor
    assert type(img) == torch.FloatTensor
    print 'Test passed: test_soft_bins'


def test_init_gmm():
    # Pass paths to cached GMM and mean Lightness
    GMM_PATH = '/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/gmm.pkl'
    MEAN_L_PATH = '/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/mean_l.npy'
    dataset = \
        data_loader.ColorizeImageNet(
            root, split='train', set='tiny', bins='soft', 
            gmm_path=GMM_PATH, mean_l_path=MEAN_L_PATH)
    print 'Test passed: test_init_gmm'



def main():
    test_single_read()
    test_single_read_dimcheck()
    test_train_loader()
    test_cmyk_read()
    test_grayscale_read()
    test_soft_bins()
    test_lowpass_image()
    test_init_gmm()

    # 
    # dataset.get_color_samples()
    # test_dataset_read()
    # TODO - test_labels
    # TODO - test colorspace conversions





if __name__ == '__main__':
    main()
