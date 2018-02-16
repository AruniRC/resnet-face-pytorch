import argparse
import os
import os.path as osp

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt

here = osp.dirname(osp.abspath(__file__)) # output folder is located here
root_dir,_ = osp.split(here)
import sys
sys.path.append(root_dir)



'''
Calculate mean R, G, B values for VGGFace2 dataset
--------------------------------------------------
Following implementation: [VGGFace2](https://arxiv.org/pdf/1710.08092.pdf)
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', 
                        default='/srv/data1/arunirc/datasets/vggface2')
    args = parser.parse_args()


    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size

    # -----------------------------------------------------------------------------
    # 1. Dataset
    # ----------------------------------------------------------------------------- 
    data_root = args.dataset_path
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Scale(256),  # smaller side resized
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    # Data loaders
    traindir = osp.join(data_root, 'train')
    dataset = datasets.ImageFolder(traindir, train_transform)
    train_loader = torch.utils.data.DataLoader(
                    dataset, shuffle=True, batch_size=128, **kwargs)
    
    rgb_mean = []
    for batch_idx, (images, lbl) in tqdm.tqdm( enumerate(train_loader), 
                                        total=len(train_loader), 
                                        desc='Sampling images' ): 
        rgb_mean.append( ((images.mean(dim=0)).mean(dim=-1)).mean(dim=-1) )
        if batch_idx == 100:
            break

    print len(rgb_mean)
    rgb_mean = torch.mean( torch.stack(rgb_mean, dim=1), dim=1)
    print rgb_mean

    res = {}
    res['R'] = rgb_mean[0]
    res['G'] = rgb_mean[1]
    res['B'] = rgb_mean[2]

    with open(osp.join(here, 'mean_rgb.yaml'), 'w') as f:
        yaml.dump(res, f, default_flow_style=False)




if __name__ == '__main__':
    main()

