from __future__ import division

import math
import warnings

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import scipy.ndimage
import six
import skimage
import skimage.color
from skimage import img_as_ubyte
import os
import os.path as osp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt





def make_weights_for_balanced_classes(images, nclasses):  
    '''
        Make a vector of weights for each image in the dataset, based 
        on class frequency. The returned vector of weights can be used 
        to create a WeightedRandomSampler for a DataLoader to have 
        class balancing when sampling for a training batch. 
            images - torchvisionDataset.imgs 
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3                      
    '''
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))  # total number of images                  
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight


def get_vgg_class_counts(log_path):
    '''  Dict of class frequencies from pre-computed text file  '''
    data_1 = np.genfromtxt(log_path, dtype=None)
    class_names = [x[0] for x in data_1]
    class_counts = [x[1] for x in data_1]
    class_count_dict = dict(zip(class_names, class_counts))
    return class_count_dict



def plot_log_csv(log_path):
    log_dir, _ = osp.split(log_path)
    dat = np.genfromtxt(log_path, names=True, 
                        delimiter=',', autostrip=True)

    train_loss =  dat['trainloss']
    train_loss_sel = ~np.isnan(train_loss)
    train_loss = train_loss[train_loss_sel]
    iter_train_loss = dat['iteration'][train_loss_sel]

    train_acc = dat['trainacc']
    train_acc_sel = ~np.isnan(train_acc)
    train_acc = train_acc[train_acc_sel]
    iter_train_acc = dat['iteration'][train_acc_sel]

    val_loss =  dat['validloss']
    val_loss_sel = ~np.isnan(val_loss)
    val_loss = val_loss[val_loss_sel]
    iter_val_loss = dat['iteration'][val_loss_sel]

    mean_iu = dat['validacc']
    mean_iu_sel = ~np.isnan(mean_iu)
    mean_iu = mean_iu[mean_iu_sel]
    iter_mean_iu = dat['iteration'][mean_iu_sel]

    fig, ax = plt.subplots(nrows=2, ncols=2)

    plt.subplot(2, 2, 1)
    plt.plot(iter_train_acc, train_acc, label='train')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.plot(iter_mean_iu, mean_iu, label='val')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.plot(iter_train_loss, train_loss, label='train')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.plot(iter_val_loss, val_loss, label='val')
    plt.xlabel('iteration')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(osp.join(log_dir, 'log_plots.png'), bbox_inches='tight')

