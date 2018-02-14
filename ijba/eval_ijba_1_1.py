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


'''
Evaluate a network on the IJB-A verification task
=================================================
Example usage: TODO *** 
# Resnet 101 on 10 folds of IJB-A 1:1
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', default='lfw_eval')
    parser.add_argument('-g', '--gpu', type=int, default=0)

    parser.add_argument('-d', '--data_dir', 
                        default='/home/renyi/arunirc/data1/datasets/CS2')
    parser.add_argument('-p', '--protocol_dir', 
                        default='/home/renyi/arunirc/data1/datasets/IJB-A/IJB-A_11_sets/')
    parser.add_argument('--fold', type=int, default=1, choices=[1,10])

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-m', '--model_path', 
                        default='/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_bottleneck_ft2_CFG-023_TIME-20180213-091016/model_best.pth.tar', 
                        required=False, # TODO - debugging...
                        help='Path to pre-trained model')
    parser.add_argument('--model_type', default='resnet101-512d',
                        choices=['resnet50', 'resnet101', 'resnet101-512d'])
    
    args = parser.parse_args()


    # CUDA setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size    

    # -----------------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------------
    fold_id = 1
    file_ext = '.jpg'
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

    pairs_path = osp.join(args.protocol_dir, 'split%d' % fold_id, 
                          'verify_comparisons_%d.csv' % fold_id)
    pairs = utils.read_ijba_pairs(pairs_path)
    protocol_file = osp.join(args.protocol_dir, 'split%d' % fold_id, 
                          'verify_metadata_%d.csv' % fold_id)
    metadata = utils.get_ijba_1_1_metadata(protocol_file) # dict
    assert np.all(np.unique(pairs) == np.unique(metadata['template_id']))  # sanity-check
    path_list = np.array([osp.join(args.data_dir, str(x)+file_ext) 
                         for x in metadata['sighting_id'] ]) # face crops saved as <sighting_id.jpg>

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
                        data_loader.IJBADataset(
                        path_list, test_transform, split=fold_id), 
                        batch_size=args.batch_size, shuffle=False )

    # testing
    # for i in range(len(test_loader.dataset)):
    #     img = test_loader.dataset.__getitem__(i)
    #     sz = img.shape
    #     if sz[0] != 3:
    #         print sz


    # -----------------------------------------------------------------------------
    # 2. Model
    # -----------------------------------------------------------------------------
    num_class = 8631
    if args.model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(2048, num_class)
    elif args.model_type == 'resnet101':
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(2048, num_class)
    elif args.model_type == 'resnet101-512d':
        model = torchvision.models.resnet101(pretrained=False)
        layers = []
        layers.append(torch.nn.Linear(2048, 512))
        layers.append(torch.nn.Linear(512, num_class))
        model.fc = torch.nn.Sequential(*layers)
    else:
        raise NotImplementedError
    
    checkpoint = torch.load(args.model_path)       
    if checkpoint['arch'] == 'DataParallel':
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.module # get network module from inside its DataParallel wrapper
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if cuda:
        model = model.cuda()

    # Convert the trained network into a "feature extractor"
    feature_map = list(model.children())
    if args.model_type == 'resnet101-512d':
        model.eval()
        extractor = model
        extractor.fc = nn.Sequential(extractor.fc[0])
    else: 
        feature_map.pop()
        extractor = nn.Sequential(*feature_map)
    
    extractor.eval() # set to evaluation mode (fixes BatchNorm, dropout, etc.)


    # -----------------------------------------------------------------------------
    # 3. Feature extraction
    # -----------------------------------------------------------------------------
    features = []
    for batch_idx, images in tqdm.tqdm(enumerate(test_loader), 
                                        total=len(test_loader), 
                                        desc='Extracting features'): 
        x = Variable(images, volatile=True) # test-time memory conservation
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
    for pair in pairs:

        import pdb; pdb.set_trace()  # breakpoint 7e2d9783 //

        print pair







if __name__ == '__main__':
    main()

