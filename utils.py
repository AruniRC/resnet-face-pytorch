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


def plot_log_csv(log_path):
	log_dir, _ = osp.split(log_path)
	dat = np.genfromtxt(log_path, names=True, 
						delimiter=',', autostrip=True)

	# train_loss =  dat['trainloss']
	# train_loss_sel = ~np.isnan(train_loss)
	# train_loss = train_loss[train_loss_sel]
	# iter_train_loss = dat['iteration'][train_loss_sel]

	val_loss =  dat['validloss']
	val_loss_sel = ~np.isnan(val_loss)
	val_loss = val_loss[val_loss_sel]
	iter_val_loss = dat['iteration'][val_loss_sel]

	mean_iu = dat['validmean_iu']
	mean_iu_sel = ~np.isnan(mean_iu)
	mean_iu = mean_iu[mean_iu_sel]
	iter_mean_iu = dat['iteration'][mean_iu_sel]

	f = plt.figure()
	plt.plot(iter_mean_iu, mean_iu, label='val')
	plt.xlabel('iteration')
	plt.ylabel('mean IoU')
	plt.grid()
	plt.legend()
	plt.savefig(osp.join(log_dir, 'val_mean_iou.png'), bbox_inches='tight')

	f = plt.figure()
	plt.plot(iter_val_loss, val_loss, label='val')
	plt.xlabel('iteration')
	plt.ylabel('KLdiv loss')
	plt.grid()
	plt.legend()
	plt.savefig(osp.join(log_dir, 'val_loss.png'), bbox_inches='tight')



def colorize_image_hc(labels, img, gmm, mean_l):
    '''
        Colorizes a grayscale image given the colorizer network 
        predictions for joint Hue/Chroma cluster centroids. 

        Inputs
        ------
        labels   -   Predicted labels as float numpy array 
        img      -   Grayscale image as float numpy array
        gmm      -   GMM model of Hue/Chroma

        Output
        ------
        im_out   -   RGB image

        Example
        -------
        from sklearn.externals import joblib
        import numpy as np
        import utils
        gmm_path = osp.join(log_dir, 'gmm.pkl')
        gmm = joblib.load(gmm_path)
        mean_l_path = osp.join(log_dir, 'mean_l.npy')
        mean_l = np.load(mean_l_path)
        # ... predicted labels and input image (mean subtracted)
        labels = labels.numpy()
        img = img.squeeze().numpy()
        im_rgb = utils.colorize_image_hc(labels, img, gmm, mean_l)
    '''
    labels = labels.astype(gmm.means_.dtype)
    img = img.astype(gmm.means_.dtype)

    # expectation over GMM centroids
    hc_means = gmm.means_.astype(labels.dtype)
    im_hc = np.tensordot(labels, hc_means, (2,0)) 
    im_l = img + mean_l.astype(img.dtype)
    im_rgb = _hue_chroma_to_rgb(im_hc, im_l)
    low, high = np.min(im_rgb), np.max(im_rgb)
    im_rgb = (im_rgb - low) / (high - low)
    im_out = img_as_ubyte(im_rgb)
    return im_out


def _rgb_to_hue_chroma(img):
    im_hsv = skimage.color.rgb2hsv(img)
    if np.isnan(np.sum(im_hsv)):
        raise ValueError('HSV from RGB conversion has NaN.')
    h = im_hsv[:,:,0]
    s = im_hsv[:,:,1]
    v = im_hsv[:,:,2]
    c = v * s
    L = v - (c/2.0)
    return h, c, L


def _hue_chroma_to_rgb(im_hc, im_l):
    h = im_hc[:,:,0]
    c = im_hc[:,:,1]
    v = im_l + (c/2.0)
    s = c/v
    im_hsv = np.stack((h,s,v), axis=2)
    im_rgb = skimage.color.hsv2rgb(im_hsv)
    return im_rgb




def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc



def visualize_segmentation(lbl_pred, lbl_true, img, im_l, \
	n_class, viz_type='avg'):
    '''
        Returns a visualization of predictions and ground-truth labels
        [rgb_img, true_labels | grayscale_img, pred_labels]
        viz_type - {'random', 'avg'}
    '''
    assert len(img.shape)==3
    assert len(im_l.shape)==2
    assert len(lbl_pred.shape)==2
    assert len(lbl_true.shape)==2

    if n_class < len(skimage.color.color_dict.keys()):
        color_map = skimage.color.color_dict.keys()[:n_class]
    else:
        color_map = skimage.color.color_dict.keys()

    img = skimage.img_as_float(img)
    if viz_type=='avg':
        im_lbl_true = skimage.color.label2rgb(lbl_true, img, kind='avg')
        im_lbl_pred = skimage.color.label2rgb(lbl_pred, img, kind='avg')
    elif viz_type=='random':
        im_lbl_true = skimage.color.label2rgb(lbl_true, img, 
                                                colors=color_map, alpha=0.7)
        im_lbl_pred = skimage.color.label2rgb(lbl_pred, img, 
                                                colors=color_map, alpha=0.7)
    im_gray_rgb = skimage.color.gray2rgb(im_l)

    tiled_img = np.concatenate(
                (np.zeros([img.shape[0],10,3]), 
                 img, im_lbl_true, 
                 np.zeros([img.shape[0],10,3]),
                 im_gray_rgb, im_lbl_pred, 
                 np.zeros([img.shape[0],10,3])), 
                axis=1)
    return skimage.img_as_ubyte(tiled_img)



# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
# From: https://github.com/wkentaro/fcn/blob/master/fcn/utils.py
def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size
    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical + h,
               pad_horizontal:pad_horizontal + w] = src
    return centerized


def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.
    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, 3), dtype=np.uint8)
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in six.moves.range(y_num):
        for x in six.moves.range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y * one_height:(y + 1) * one_height,
                                   x * one_width:(x + 1) * one_width] = imgs[i]
    return concatenated_image


def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None):
    """Concatenate images whose sizes are different.
    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """
    from skimage.transform import resize

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(math.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return x_num, y_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_height, max_width = np.inf, np.inf
    for img in imgs:
        max_height = min([max_height, img.shape[0]])
        max_width = min([max_width, img.shape[1]])

    # resize and concatenate images
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        dtype = img.dtype
        h_scale, w_scale = max_height / h, max_width / w
        scale = min([h_scale, w_scale])
        h, w = int(scale * h), int(scale * w)
        img = resize(img, (h, w), preserve_range=True).astype(dtype)
        if len(img.shape) == 3:
            img = centerize(img, (max_height, max_width, 3), margin_color)
        else:
            img = centerize(img, (max_height, max_width), margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)
