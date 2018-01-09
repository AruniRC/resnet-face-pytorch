import argparse
import os
import os.path as osp

# import torch
# import torchvision
# import torch.utils.data
# import torchvision.datasets as datasets

import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm
# from multiprocessing.pool import ThreadPool as Pool
# pool_size = 5  # your "parallelness"
# pool = Pool(pool_size)

'''
    Crops out the faces from UMDFace images using the annotations in 
    umdfaces_batch*_ultraface.csv. 
    Automatically creates "train" and "val" folders.

'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', 
                        default='/srv/data1/arunirc/datasets/UMDFaces/', 
                        help='Location of the folders containing 3 batches of UMDFaces stills.')
    parser.add_argument('-o', '--output_path', 
                        default='/srv/data1/arunirc/datasets/UMDFaces/face_crops')
    parser.add_argument('-n', '--num_val', type=int, default=2)
    parser.add_argument('-b', '--batch', type=int, default=-1,
                        help='crop faces of specified UMDFaces batch')
    args = parser.parse_args()

    # torch.manual_seed(1337)

    if not osp.exists(args.output_path):
        os.makedirs(args.output_path)   

    if not osp.exists(osp.join(args.output_path, 'train')):
        os.makedirs(osp.join(args.output_path, 'train')) 

    if not osp.exists(osp.join(args.output_path, 'val')):
        os.makedirs(osp.join(args.output_path, 'val'))


    # -----------------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------------
    dir_batch = (
                 osp.join(args.dataset_path, 'umdfaces_batch1'),
                 osp.join(args.dataset_path, 'umdfaces_batch2'),
                 osp.join(args.dataset_path, 'umdfaces_batch3'))

    # dataset_batch = [datasets.ImageFolder(b) for b in dir_batch]

    annot_files = (
                   osp.join(dir_batch[0], 'umdfaces_batch1_ultraface.csv'),
                   osp.join(dir_batch[1], 'umdfaces_batch2_ultraface.csv'),
                   osp.join(dir_batch[2], 'umdfaces_batch3_ultraface.csv'))

    for fn in annot_files:
        assert osp.exists(fn)

    if args.batch < 0:
        # by default loop over batches in order
        for i in range(len(dir_batch)):
             crop_batch(dir_batch[i], annot_files[i], 
                        args.output_path, args.num_val)
    else:
        i = args.batch 
        crop_batch(dir_batch[i], annot_files[i], args.output_path, args.num_val)



    # dataset_all = torch.utils.data.ConcatDataset(
    #                 (dataset_batch1, dataset_batch2, dataset_batch3))
    # for i in range(100):
    #     pool.apply_async(f, (item,))


def crop_batch(data_dir, annot_fn, out_dir, nval):

    dat = np.genfromtxt(annot_fn, names=True, delimiter=',', 
                        autostrip=True, dtype=None)
    im_fn = dat['FILE']
    (face_x, face_y, face_w, face_h) = (
                                        dat['FACE_X'],
                                        dat['FACE_Y'],
                                        dat['FACE_WIDTH'],
                                        dat['FACE_HEIGHT'])

    class_ids = dat['SUBJECT_ID']

    for c in tqdm(range(len(class_ids))):
        sel = (class_ids==class_ids[c])

        class_image_fn = im_fn[sel]

        for i in xrange(len(class_image_fn)):
            # print class_image_fn[i]
            im = PIL.Image.open(osp.join(data_dir, class_image_fn[i]))
            rect = (face_x[sel][i], face_y[sel][i], 
                    face_x[sel][i]+ face_w[sel][i], 
                    face_y[sel][i]+face_h[sel][i])        
            imc = im.crop(rect)
            
            class_name, _ = osp.split(class_image_fn[i])

            if not osp.exists(osp.join(out_dir, 'train', class_name)):
                os.makedirs(osp.join(out_dir, 'train', class_name))

            if i < len(class_image_fn)-nval:
                imc.save(osp.join(out_dir, 'train', class_image_fn[i]))
            else:
                if not osp.exists(osp.join(out_dir, 'val', class_name)):
                    os.makedirs(osp.join(out_dir, 'val', class_name))
                imc.save(osp.join(out_dir, 'val', class_image_fn[i]))




if __name__ == '__main__':
    main()

