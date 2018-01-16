
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import os
import os.path as osp
import yaml
import numpy as np
import PIL.Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils




here = osp.dirname(osp.abspath(__file__)) # output folder is located here

    
# -----------------------------------------------------------------------------
# 0. User-defined settings
# -----------------------------------------------------------------------------
gpu = 0 # use gpu:0 by default
# specify model path of trained ResNet-50 network:
model_path = './umd-face/logs/MODEL-resnet_umdfaces_CFG-006_TIME-20180114-141943/model_best.pth.tar'  
num_class = 8277 # UMD-Faces had this many classes



# -----------------------------------------------------------------------------
# 1. GPU setup
# -----------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
cuda = torch.cuda.is_available()
torch.manual_seed(1337)
if cuda:
    torch.cuda.manual_seed(1337)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True    



# -----------------------------------------------------------------------------
# 2. Data preparation
# -----------------------------------------------------------------------------

# Samples images taken for demo purpose from LFW:
#   http://vis-www.cs.umass.edu/lfw/
data_root = './samples/verif'
file_path = [osp.join(data_root, 'Recep_Tayyip_Erdogan_0012.jpg'), 
             osp.join(data_root, 'Recep_Tayyip_Erdogan_0015.jpg'), 
             osp.join(data_root, 'Quincy_Jones_0001.jpg')]
image = [PIL.Image.open(f).convert('RGB') for f in file_path]

# Data transforms
# http://pytorch.org/docs/master/torchvision/transforms.html
# NOTE: these should be consistent with the training script val_loader
# Since LFW images are 250x250 and not close-crops, we modify the cropping a bit.
RGB_MEAN = [ 0.485, 0.456, 0.406 ]
RGB_STD = [ 0.229, 0.224, 0.225 ]
test_transform = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.Scale((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = RGB_MEAN,
                         std = RGB_STD),
])

# apply the transform
inputs = [test_transform(im) for im in image]

# # Visualize
# for im in image:
#     im.show() # non-blocking display of PIL Images



# -----------------------------------------------------------------------------
# 2. Model
# -----------------------------------------------------------------------------
# PyTorch ResNet model definition: 
#   https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# ResNet docs:
#   http://pytorch.org/docs/master/torchvision/models.html#id3
model = torchvision.models.resnet50(pretrained=True) # Using pre-trained for demo purpose

# Replace last layer (by default, resnet has 1000 output categories)
model.fc = torch.nn.Linear(2048, num_class) # change to current dataset's classes

# Pre-trained PyTorch model loaded from a file
checkpoint = torch.load(model_path)        

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
# From https://github.com/meliketoy/fine-tuning.pytorch/blob/master/extract_features.py#L85
feature_map = list(model.children())
feature_map.pop()  # remove the final "class prediction" layer
extractor = nn.Sequential(*feature_map) # create feature extractor

# Inspect the structure - it is a nested list of various modules
print extractor[-1]      # last layer of the model - avg-pool
print extractor[-2][-1]  # second-last layer's last module - output is 2048-dim



# -----------------------------------------------------------------------------
# 3. Feature extraction
# -----------------------------------------------------------------------------
# - simple, one input sample at a time
features = []
for x in inputs:
    x = Variable(x, volatile=True)
    if cuda:
        x = x.cuda()
    x = x.view(1, x.size(0), x.size(1), x.size(2)) # add batch_dim=1 in the front
    feat = extractor(x).view(-1) # extract features of input `x`, reshape to vector
    features.append(feat)
features = torch.stack(features) # N x 2048 for N inputs

# get Tensors on CPU from autograd.Variables on GPU
if cuda:
    features = features.data.cpu()
else:
    features = features.data



# -----------------------------------------------------------------------------
# 4. Face verification - TODO
# -----------------------------------------------------------------------------
features = F.normalize(features, p=2, dim=1)  # L2-normalize

# L2-distance between features (Tensors) of same and different pairs
d1 = (features[0] - features[1]).norm(p=2) # same pair
d2 = (features[0] - features[2]).norm(p=2) # different pair

print 'matched pair: %.2f' % d1   
print 'mismatched pair: %.2f' % d2
assert d1 < d2

# visualizations
fig, ax = plt.subplots(nrows=2, ncols=2)
plt.subplot(2, 2, 1)
plt.title('matched pair')
plt.imshow(image[0])
plt.tight_layout()

plt.subplot(2, 2, 2)
plt.imshow(image[1])
plt.title('d = %.3f' % d1)
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.imshow(image[0])
plt.title('mismatched pair')
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.imshow(image[2])
plt.title('d = %.3f' % d2)
plt.tight_layout()

# plt.tight_layout()

plt.savefig(osp.join(here, 'demo_verif.png'), bbox_inches='tight')

print 'Visualization saved in: ' + osp.join(here, 'demo_verif.png')
