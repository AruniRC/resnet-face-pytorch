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

import yaml
import tqdm
import numpy as np
import sklearn.metrics
from sklearn import metrics
from scipy import interpolate
import scipy.io as sio
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
Evaluate a network on the IJB-A 1:1 verification task
=====================================================
Example usage: TODO *** 
# Resnet 101 on 10 folds of IJB-A 1:1
'''
# MODEL_PATH = '/srv/data1/arunirc/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_vggface_scratch_CFG-022_TIME-20180210-201442/model_best.pth.tar'


# Resnet101-512d-norm 
# MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_512d_L2norm_ft2_CFG-023_TIME-20180214-020054/model_best.pth.tar'
MODEL_TYPE = 'resnet101-512d-norm'
# MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_512d_L2norm_ft2_CFG-022_TIME-20180214-015313/model_best.pth.tar'
MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_512d_L2norm_ft2_CFG-024_TIME-20180214-160410/model_best.pth.tar'


# Resnet101-512d
# MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_bottleneck_ft2_CFG-023_TIME-20180213-091016/model_best.pth.tar'
# MODEL_TYPE = 'resnet101-512d'
# MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_bottleneck_ft1_CFG-021_TIME-20180212-192332/model_best.pth.tar'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', default='ijba_eval')
    parser.add_argument('-g', '--gpu', type=int, default=0)

    parser.add_argument('-d', '--data_dir', 
                        default='/home/renyi/arunirc/data1/datasets/CS2')
    parser.add_argument('-p', '--protocol_dir', 
                        default='/home/renyi/arunirc/data1/datasets/IJB-A/IJB-A_11_sets/')
    parser.add_argument('--fold', type=int, default=1, choices=[1,10])
    parser.add_argument('--sqrt', action='store_true', default=False,
                        help='Add signed sqrt normalization')
    parser.add_argument('--cosine', action='store_true', default=False,
                        help='Use cosine similarity instead of L2 distance')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-m', '--model_path', 
                        default=MODEL_PATH, 
                        help='Path to pre-trained model')
    parser.add_argument('--model_type', default=MODEL_TYPE,
                        choices=['resnet50', 'resnet101', 'resnet101-512d', 'resnet101-512d-norm'])
    
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
    # 1. Model
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
    elif args.model_type == 'resnet101-512d-norm':
        model = torchvision.models.resnet101(pretrained=False)
        layers = []
        layers.append(torch.nn.Linear(2048, 512))
        layers.append(models.NormFeat(scale_factor=50.0))
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
    if args.model_type == 'resnet101-512d' or args.model_type == 'resnet101-512d-norm':
        model.eval()
        extractor = model
        extractor.fc = nn.Sequential(extractor.fc[0])
    else: 
        feature_map.pop()
        extractor = nn.Sequential(*feature_map)
    extractor.eval() # ALWAYS set to evaluation mode (fixes BatchNorm, dropout, etc.)



    # -----------------------------------------------------------------------------
    # 2. Dataset
    # -----------------------------------------------------------------------------
    fold_id = 1
    file_ext = '.jpg'
    RGB_MEAN = [ 0.485, 0.456, 0.406 ]
    RGB_STD = [ 0.229, 0.224, 0.225 ]

    test_transform = transforms.Compose([
        # transforms.Scale(224),
        # transforms.CenterCrop(224),
        transforms.Scale((224,224)),
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
    # 3. Feature extraction
    # -----------------------------------------------------------------------------
    print 'Feature extraction...'
    cache_dir = osp.join(here, 'cache-' + args.model_type)
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir)

    feat_path = osp.join(cache_dir, 'feat-fold-%d.mat' % fold_id)

    if not osp.exists(feat_path):
        features = []
        for batch_idx, images in tqdm.tqdm(enumerate(test_loader), 
                                            total=len(test_loader), 
                                            desc='Extracting features'): 
            x = Variable(images, volatile=True) # test-time memory conservation
            if cuda:
                x = x.cuda()
            feat = extractor(x)
            if cuda:
                feat = feat.data.cpu() # free up GPU
            else:
                feat = feat.data
            features.append(feat)

        features = torch.cat(features, dim=0) # (n_batch*batch_sz) x 512
        sio.savemat(feat_path, {'feat': features.cpu().numpy() })
    else:
        dat = sio.loadmat(feat_path)
        features = torch.FloatTensor(dat['feat'])
        del dat
        print 'Loaded.'


    # -----------------------------------------------------------------------------
    # 4. Verification
    # -----------------------------------------------------------------------------
    scores = []
    labels = []

    # labels: is_same_subject
    print 'Computing pair labels . . . '
    for pair in tqdm.tqdm(pairs): # TODO - check tqdm
        sel_t0 = np.where(metadata['template_id'] == pair[0])
        sel_t1 = np.where(metadata['template_id'] == pair[1])
        subject0 = np.unique(metadata['subject_id'][sel_t0])
        subject1 = np.unique(metadata['subject_id'][sel_t1])
        labels.append(int(subject0 == subject1))
    labels = np.array(labels)
    print 'done'

    # templates: average pool, then L2-normalize
    print 'Pooling templates . . . '
    pooled_features = []
    template_set = np.unique(metadata['template_id'])
    for tid in tqdm.tqdm(template_set):
        sel = np.where(metadata['template_id'] == tid)
        # pool template: 1 x n x 512 -> 1 x 512
        feat = features[sel,:].mean(1)
        if args.sqrt:  # signed-square-root normalization
            feat = torch.mul(torch.sign(feat),torch.sqrt(torch.abs(feat)+1e-12))
        pooled_features.append(F.normalize(feat, p=2, dim=1) )    
    pooled_features = torch.cat(pooled_features, dim=0) # (n_batch*batch_sz) x 512
    print 'done'

    print 'Computing pair distances . . . '
    for pair in tqdm.tqdm(pairs):
        sel_t0 = np.where(template_set == pair[0])
        sel_t1 = np.where(template_set == pair[1])
        if args.cosine:
            feat_dist = torch.dot(torch.squeeze(pooled_features[sel_t0]), 
                                  torch.squeeze(pooled_features[sel_t1]))
        else:
            feat_dist = (pooled_features[sel_t0] - pooled_features[sel_t1]).norm(p=2, dim=1)
            feat_dist = -torch.squeeze(feat_dist)
            feat_dist = feat_dist.numpy()
        scores.append(feat_dist) # score: negative of L2-distance
    scores = np.array(scores)

    # Metrics: TAR (tpr) at FAR (fpr)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)
    fpr_levels = [0.0001, 0.001, 0.01, 0.1]
    f_interp = interpolate.interp1d(fpr, tpr)
    tpr_at_fpr = [ f_interp(x) for x in fpr_levels ]

    for (far, tar) in zip(fpr_levels, tpr_at_fpr):
        print 'TAR @ FAR=%.4f : %.4f' % (far, tar)

    res = {}
    res['TAR'] = tpr_at_fpr
    res['FAR'] = fpr_levels
    with open( osp.join(cache_dir, 'result-1-1-fold-%d.yaml' % fold_id), 
              'w') as f:
        yaml.dump(res, f, default_flow_style=False)

    sio.savemat(osp.join(cache_dir, 'roc-1-1-fold-%d.mat' % fold_id), 
                {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 
                    'tpr_at_fpr': tpr_at_fpr})


if __name__ == '__main__':
    main()

