import datetime
import math
import os
import os.path as osp
import shutil

import numpy as np
import PIL.Image
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import utils
import gc


def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer



class Trainer(object):

    # -----------------------------------------------------------------------------
    def __init__(self, cuda, model, criterion, optimizer,
                 train_loader, val_loader, out, max_iter, interval_validate=None):
    # -----------------------------------------------------------------------------
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_acc = 0

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('US/Eastern'))

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'valid/loss',
            'valid/acc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_loss = 0


    # -----------------------------------------------------------------------------
    def validate(self, max_num=500):
    # -----------------------------------------------------------------------------
        training = self.model.training
        self.model.eval()
        MAX_NUM = max_num # HACK: stop after 500 images

        n_class = len(self.val_loader.dataset.classes)
        val_loss = 0
        label_trues, label_preds = [], []

        for batch_idx, (data, (target)) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Val=%d' % self.iteration, ncols=80,
                leave=False):

            # Computing val losses
            if self.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            score = self.model(data)
            loss = self.criterion(score, target)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is NaN while validating')

            val_loss += float(loss.data[0]) / len(data)

            lbl_pred = score.data.max(1)[1].cpu().numpy()
            lbl_true = target.data.cpu()
            lbl_pred = lbl_pred.squeeze()
            lbl_true = np.squeeze(lbl_true.numpy())

            del target, score

            label_trues.append(lbl_true)
            label_preds.append(lbl_pred)

            del lbl_true, lbl_pred, data, loss

            if batch_idx > MAX_NUM:
                break

        # Computing metrics
        val_loss /= len(self.val_loader)
        val_acc = self.eval_metric(label_trues, label_preds)

        # Logging
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('US/Eastern')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 2 + \
                  [val_loss, val_acc] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        del label_trues, label_preds

        # Saving the best performing model
        is_best = val_acc > self.best_acc
        if is_best:
            self.best_acc = val_acc

        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
        }, osp.join(self.out, 'checkpoint.pth.tar'))

        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()



    # -----------------------------------------------------------------------------
    def train_epoch(self):
    # -----------------------------------------------------------------------------
        self.model.train()
        n_class = len(self.train_loader.dataset.classes)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.train_loader)

            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            # Computing Losses
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            score = self.model(data)  # batch_size x num_class

            loss = self.criterion(score, target)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is NaN while training')
            # print list(self.model.parameters())[0].grad

            # Gradient descent
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Computing metrics
            lbl_pred = score.data.max(1)[1].cpu().numpy()
            lbl_pred = lbl_pred.squeeze()
            lbl_true = target.data.cpu()
            lbl_true = np.squeeze(lbl_true.numpy())
            train_accu = self.eval_metric([lbl_pred], [lbl_true])

            # Logging
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('US/Eastern')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + \
                      [train_accu] + [''] * 2 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')
                # print '\nEpoch: ' + str(self.epoch) + ' Iter: ' + str(self.iteration) + \
                #         ' Loss: ' + str(loss.data[0])

            if self.iteration >= self.max_iter:
                break


    # -----------------------------------------------------------------------------
    def eval_metric(self, lbl_pred, lbl_true):
    # -----------------------------------------------------------------------------
        # Over-all accuracy
        # TODO: per-class accuracy
        accu = []
        for lt, lp in zip(lbl_true, lbl_pred):
            accu.append(np.mean(lt == lp))
        return np.mean(accu)


    # -----------------------------------------------------------------------------
    def train(self):
    # -----------------------------------------------------------------------------
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

