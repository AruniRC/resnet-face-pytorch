import argparse
import os
import os.path as osp

import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F

import tqdm
import numpy as np
import sklearn.metrics
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = osp.dirname(osp.abspath(__file__)) # output folder is located here
root_dir,_ = osp.split(here)
import sys
sys.path.append(root_dir)

import models
import utils
import data_loader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', default='lfw_eval_dev')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-d', '--dataset_path', 
                        default='/srv/data1/arunirc/datasets/lfw-deepfunneled')
    parser.add_argument('--fold', type=int, default=0, choices=[0,10])
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('-m', '--model_path', default=None, 
                        help='Path to pre-trained model')
    
    args = parser.parse_args()


    # CUDA setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size    

    if args.fold == 0:
        pairs_path = './lfw/data/pairsDevTest.txt'
    else:
        pairs_path = './lfw/data/pairs.txt'

    # -----------------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------------
    file_ext = 'jpg' # observe, no '.' before jpg
    num_class = 8631

    pairs = utils.read_pairs(pairs_path)
    path_list, issame_list = utils.get_paths(args.dataset_path, pairs, file_ext)

    # Define data transforms
    RGB_MEAN = [ 0.485, 0.456, 0.406 ]
    RGB_STD = [ 0.229, 0.224, 0.225 ]
    test_transform = transforms.Compose([
        transforms.Scale((250,250)),  # make 250x250
        transforms.CenterCrop(150),   # then take 150x150 center crop
        transforms.Scale((224,224)),  # resized to the network's required input size
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
                        data_loader.LFWDataset(
                        path_list, issame_list, test_transform), 
                        batch_size=args.batch_size, shuffle=False )


    # -----------------------------------------------------------------------------
    # 2. Model
    # -----------------------------------------------------------------------------
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, num_class)
    checkpoint = torch.load(args.model_path)        

    if checkpoint['arch'] == 'DataParallel':
        # if we trained and saved our model using DataParallel
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.module # get network module from inside its DataParallel wrapper
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if cuda:
        model = model.cuda()

    # Convert the trained network into a "feature extractor"
    feature_map = list(model.children())
    feature_map.pop()  # remove the final "class prediction" layer
    extractor = nn.Sequential(*feature_map) # create feature extractor



    # -----------------------------------------------------------------------------
    # 3. Feature extraction
    # -----------------------------------------------------------------------------
    features = []

    for batch_idx, images in tqdm.tqdm(enumerate(test_loader), 
                                        total=len(test_loader), 
                                        desc='Extracting features'): 
        x = Variable(images, volatile=True)
        if cuda:
            x = x.cuda()
        feat = extractor(x)
        if cuda:
            feat = feat.data.cpu()
        else:
            feat = feat.data
        features.append(feat)

    features = torch.stack(features)
    sz = features.size()
    features = features.view(sz[0]*sz[1], sz[2])
    features = F.normalize(features, p=2, dim=1) # L2-normalize


    # -----------------------------------------------------------------------------
    # 4. Verification
    # -----------------------------------------------------------------------------
    num_feat = features.size()[0]
    feat_pair1 = features[np.arange(0,num_feat,2),:]
    feat_pair2 = features[np.arange(1,num_feat,2),:]
    feat_dist = (feat_pair1 - feat_pair2).norm(p=2, dim=1)
    feat_dist = feat_dist.numpy()

    # Eval metrics
    scores = -feat_dist
    gt = np.asarray(issame_list)
    
    # TODO - EER    
    if args.fold == 0:
        fig_path = osp.join(here, 'lfw_roc_devTest.png')
        roc_auc = sklearn.metrics.roc_auc_score(gt, scores)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt, scores)
        print 'ROC-AUC: %.04f' % roc_auc
        # Plot and save ROC curve
        fig = plt.figure()
        plt.title('ROC - lfw')
        plt.plot(fpr, tpr, lw=2, label='ROC (auc = %0.4f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.tight_layout()
    else:
        # 10 fold auc and error bars
        fold_size = 600 # 600 pairs in each fold
        roc_auc = np.zeros(10)
        roc_eer = np.zeros(10)

        fig = plt.figure()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        for i in tqdm.tqdm(range(10)):
            start = i * fold_size
            end = (i+1) * fold_size
            scores_fold = scores[start:end]
            gt_fold = gt[start:end]
            roc_auc[i] = sklearn.metrics.roc_auc_score(gt_fold, scores_fold)
            fpr, tpr, _ = sklearn.metrics.roc_curve(gt_fold, scores_fold)
            roc_eer[i] = brentq(
                            lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            plt.plot(fpr, tpr, alpha=0.4, 
                    lw=2, color='darkgreen',
                    label='ROC(auc=%0.4f, eer=%0.4f)' % (roc_auc[i], roc_eer[i]) )

        plt.title( 'AUC: %0.4f +/- %0.4f, EER: %0.4f +/- %0.4f' % 
                    (np.mean(roc_auc), np.std(roc_auc),
                     np.mean(roc_eer), np.std(roc_eer)) )

        fig_path = osp.join(here, 'lfw_roc_10fold.png')
        

    plt.savefig(fig_path, bbox_inches='tight')
    print 'ROC curve saved at: ' + fig_path






if __name__ == '__main__':
    main()

