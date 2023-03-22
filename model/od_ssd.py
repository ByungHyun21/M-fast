import os, sys, yaml, time, wandb
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from model.collection_layer import *
from model.collection_augmentation import *
from model.backbone_mobilenet_v2 import Mobilenet_v2_from_torchvision, Mobilenet_v2_from_scratch, Mobilenet_v2_extra
from model.utils import *


class ssd(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, config, ddp_rank):
        super().__init__()
        self.ddp_rank = ddp_rank
        self.model_name = config['PROJECT']
        self.nc = len(config['CLASS']) + 1 # background
        self.image_size = config['INPUT_SIZE']
        
        self.mean = torch.reshape(torch.tensor(config['MEAN'], device=self.ddp_rank), (1, -1, 1, 1))
        self.std = torch.reshape(torch.tensor(config['STD'], device=self.ddp_rank), (1, -1, 1, 1))
        
        self.FREEZE_BACKBONE = config['FREEZE_BACKBONE']

        # # # # # # # #
        # Backbone
        self.backbone = None
        
        # # # # #
        # MobileNet V2 SSD
        if config['BACKBONE'] == 'mobilenet_v2':
            if config['PRETRAINED'] == 'torchvision':
                self.backbone = Mobilenet_v2_from_torchvision()
            else:
                self.backbone = Mobilenet_v2_from_scratch(config)
                self.backbone.apply(weights_init)

            self.extra = Mobilenet_v2_extra(config)
            head_filters = [576, 1280, 512, 256, 256, 128]

        # # # # #
        # SSD head
        self.head = ssd_head(config, head_filters)
        # Backbone
        # # # # # # # #
        if config['WEIGHT'] is not None:
            pass
        
        self.extra.apply(weights_init)
        self.head.apply(weights_init)

    def forward(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        
        if self.FREEZE_BACKBONE:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        
        x = self.extra(x)
        x, anchor = self.head(x)
        
        return x, anchor
    
    def get_anchor(self):
        self.eval()
        _, anchor = self.forward(torch.rand((5, 3, self.image_size[0], self.image_size[1]), device=self.ddp_rank))
        self.train()
        
        return anchor

class ssd_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # # # # #
        # Loss Parameter
        self.alpha = 1.0
        self.SmoothL1Loss = nn.L1Loss(reduction='none')
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, pred, gt):
        n_anchors = gt.shape[1]
        # pred_cls : batch*anchors*class
        # pred_loc : batch*anchors*(cx, cy, w, h)
        pred_cls = pred[0][:, :, :-4]
        pred_loc = pred[0][:, :, -4:]
        
        # gt_cls : batch*anchors*class
        # gt_loc : batch*anchors*(cx, cy, w, h)
        gt_cls = gt[:, :, :-4]
        gt_loc = gt[:, :, -4:]
        
        neg = gt_cls[:, :, 0]
        pos = 1 - neg
        npos = torch.sum(pos, dim=1)

        # # # # #
        # Cross Entropy
        pred_cls = self.softmax(pred_cls)
        entropy = torch.sum(gt_cls * -torch.log(torch.clip(pred_cls, 1e-5, 1.0 - 1e-5)), axis=2)
        
        # # # # #
        # Hard Negative Mining
        e_pos = entropy * pos # positive entropy
        e_neg = entropy * neg # negative entropy
        
        e_neg, _ = e_neg.sort(descending=True)
        _, indices = e_neg.sort(descending=True)
        
        thres = torch.tile((npos * 3).unsqueeze(1), (1, n_anchors))
        
        e_neg = torch.where(indices < thres, e_neg, 0.0)
        
        loss_fg = torch.sum(e_pos, dim=1) 
        loss_bg = torch.sum(e_neg, dim=1)
        
        loss_cls = loss_fg + loss_bg
        # Hard Negative Mining
        # # # # #
        
        # # # # #
        # Regression Loss
        huber_loss = self.SmoothL1Loss(pred_loc, gt_loc)
        huber_loss = torch.mean(huber_loss, dim=2)
        
        loss_loc = self.alpha * torch.sum(pos * huber_loss, axis=1)
        
        # # # # #
        # Loss All
        loss_all = ((loss_cls + loss_loc) / npos).mean()
        
        loss_cls = (loss_cls / npos).mean()
        loss_loc = (loss_loc / npos).mean()
        
        return loss_all, loss_cls, loss_loc

class ssd_metric(nn.Module):
    def __init__(self):
        super().__init__()
        
    def sigmoid(self, x):
        return 1 / (1 +np.exp(-x))
    
    def softmax(self, x):
        f_x = np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)
        return f_x
        
    def forward(self, pred, gt):
        output = pred[0].detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        anchor = pred[1].detach().cpu().numpy()
        
        pred_cls = output[:, :, :-4]
        pred_loc = output[:, :, -4:]
        
        gt_cls = gt[:, :, :-4]
        gt_loc = gt[:, :, -4:]
        
        neg = gt_cls[:, :, 0]
        pos = 1 - neg
        npos = np.sum(pos, axis=1)
        nneg = np.sum(neg, axis=1)
        
        pred_cls = self.softmax(pred_cls)
        acc_cls = np.sum(np.sum(pred_cls * gt_cls, axis=2) * pos, axis=1) / npos
        acc_cls_neg = np.sum(np.sum(pred_cls * gt_cls, axis=2) * neg, axis=1) / nneg
        
        delta_cx, delta_cy, delta_w, delta_h = np.split(pred_loc, 4, axis=2)
        anchor_cx, anchor_cy, anchor_w, anchor_h = np.split(np.expand_dims(anchor, axis=0), 4, axis=2)

        box_cx = (delta_cx * anchor_w / 10.0) + anchor_cx
        box_cy = (delta_cy * anchor_h / 10.0) + anchor_cy
        box_w = np.exp(delta_w / 5.0) * anchor_w
        box_h = np.exp(delta_h / 5.0) * anchor_h
        
        pred_box = np.concatenate([box_cx, box_cy, box_w, box_h], axis=2)

        delta_cx, delta_cy, delta_w, delta_h = np.split(gt_loc, 4, axis=2)
        anchor_cx, anchor_cy, anchor_w, anchor_h = np.split(np.expand_dims(anchor, axis=0), 4, axis=2)

        box_cx = (delta_cx * anchor_w / 10.0) + anchor_cx
        box_cy = (delta_cy * anchor_h / 10.0) + anchor_cy
        box_w = np.exp(delta_w / 5.0) * anchor_w
        box_h = np.exp(delta_h / 5.0) * anchor_h
        
        gt_box = np.concatenate([box_cx, box_cy, box_w, box_h], axis=2)
        
        cx = pred_box[:, :, 0]
        cy = pred_box[:, :, 1]
        w = pred_box[:, :, 2]
        h = pred_box[:, :, 3]
        
        x11 = cx - (w / 2.0)
        x12 = cx + (w / 2.0)
        y11 = cy - (h / 2.0)
        y12 = cy + (h / 2.0)
        
        area1 = np.maximum(x12-x11, 0.0) * np.maximum(y12-y11, 0.0)
        
        cx = gt_box[:, :, 0]
        cy = gt_box[:, :, 1]
        w = gt_box[:, :, 2]
        h = gt_box[:, :, 3]
        
        x21 = cx - (w / 2.0)
        x22 = cx + (w / 2.0)
        y21 = cy - (h / 2.0)
        y22 = cy + (h / 2.0)
        
        area2 = np.maximum(x22-x21, 0.0) * np.maximum(y22-y21, 0.0)
        
        xmin = np.maximum(x11, x21)
        ymin = np.maximum(y11, y21)
        xmax = np.minimum(x12, x22)
        ymax = np.minimum(y12, y22)
        
        common = np.maximum(xmax-xmin, 0.0) * np.maximum(ymax-ymin, 0.0)
        
        ious = common / (area1 + area2 - common)
        
        acc_loc = np.sum(ious * pos, axis=1) / npos
        
        acc_fg = np.mean(acc_cls)
        acc_bg = np.mean(acc_cls_neg)
        acc_loc = np.mean(acc_loc)
        
        return acc_fg, acc_bg, acc_loc


class ssd_head(nn.Module):
    def __init__(self, config, head_filters):
        super().__init__()
        self.nc = len(config['CLASS']) + 1
        self.anchor = config['ANCHOR']
        self.scale = config['SCALE']
        bias = config['BIAS']

        self.cls1 = nn.Sequential(
            depth2d_bn_act(head_filters[0], head_filters[0], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[0], self.nc*self.anchor[0], (1, 1), None, padding=0, bias=bias)
            )

        self.cls2 = nn.Sequential(
            depth2d_bn_act(head_filters[1], head_filters[1], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[1], self.nc*self.anchor[1], (1, 1), None, padding=0, bias=bias)
            )

        self.cls3 = nn.Sequential(
            depth2d_bn_act(head_filters[2], head_filters[2], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[2], self.nc*self.anchor[2], (1, 1), None, padding=0, bias=bias)
            )

        self.cls4 = nn.Sequential(
            depth2d_bn_act(head_filters[3], head_filters[3], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[3], self.nc*self.anchor[3], (1, 1), None, padding=0, bias=bias)
            )

        self.cls5 = nn.Sequential(
            depth2d_bn_act(head_filters[4], head_filters[4], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[4], self.nc*self.anchor[4], (1, 1), None, padding=0, bias=bias)
            )

        self.cls6 = nn.Sequential(
            depth2d_bn_act(head_filters[5], head_filters[5], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[5], self.nc*self.anchor[5], (1, 1), None, padding=0, bias=bias)
            )

        self.loc1 = nn.Sequential(
            depth2d_bn_act(head_filters[0], head_filters[0], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[0], 4*self.anchor[0], (1, 1), None, padding=0, bias=bias)
            )

        self.loc2 = nn.Sequential(
            depth2d_bn_act(head_filters[1], head_filters[1], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[1], 4*self.anchor[1], (1, 1), None, padding=0, bias=bias)
            )

        self.loc3 = nn.Sequential(
            depth2d_bn_act(head_filters[2], head_filters[2], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[2], 4*self.anchor[2], (1, 1), None, padding=0, bias=bias)
            )

        self.loc4 = nn.Sequential(
            depth2d_bn_act(head_filters[3], head_filters[3], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[3], 4*self.anchor[3], (1, 1), None, padding=0, bias=bias)
            )

        self.loc5 = nn.Sequential(
            depth2d_bn_act(head_filters[4], head_filters[4], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[4], 4*self.anchor[4], (1, 1), None, padding=0, bias=bias)
            )

        self.loc6 = nn.Sequential(
            depth2d_bn_act(head_filters[5], head_filters[5], 'relu6', bias=bias),
            conv2d_bn_act(head_filters[5], 4*self.anchor[5], (1, 1), None, padding=0, bias=bias)
            )

    def forward(self, x):
        # # # # #
        # anchor        
        a1 = self.get_anchor(x[0].shape[2], x[0].shape[3], self.anchor[0], self.scale[0], self.scale[1])
        a2 = self.get_anchor(x[1].shape[2], x[1].shape[3], self.anchor[1], self.scale[1], self.scale[2])
        a3 = self.get_anchor(x[2].shape[2], x[2].shape[3], self.anchor[2], self.scale[2], self.scale[3])
        a4 = self.get_anchor(x[3].shape[2], x[3].shape[3], self.anchor[3], self.scale[3], self.scale[4])
        a5 = self.get_anchor(x[4].shape[2], x[4].shape[3], self.anchor[4], self.scale[4], self.scale[5])
        a6 = self.get_anchor(x[5].shape[2], x[5].shape[3], self.anchor[5], self.scale[5], 1.0)

        anchor = np.concatenate([a1, a2, a3, a4, a5, a6], axis=0)
        anchor = torch.tensor(anchor)

        # # # # #
        # class
        c1 = self.cls1(x[0])
        c2 = self.cls2(x[1])
        c3 = self.cls3(x[2])
        c4 = self.cls4(x[3])
        c5 = self.cls5(x[4])
        c6 = self.cls6(x[5])
        c1 = self.review(c1, self.nc)
        c2 = self.review(c2, self.nc)
        c3 = self.review(c3, self.nc)
        c4 = self.review(c4, self.nc)
        c5 = self.review(c5, self.nc)
        c6 = self.review(c6, self.nc)
        cls = torch.cat([c1, c2, c3, c4, c5, c6], dim=1)

        # # # # #
        # location
        l1 = self.loc1(x[0])
        l2 = self.loc2(x[1])
        l3 = self.loc3(x[2])
        l4 = self.loc4(x[3])
        l5 = self.loc5(x[4])
        l6 = self.loc6(x[5])
        l1 = self.review(l1, 4)
        l2 = self.review(l2, 4)
        l3 = self.review(l3, 4)
        l4 = self.review(l4, 4)
        l5 = self.review(l5, 4)
        l6 = self.review(l6, 4)
        loc = torch.cat([l1, l2, l3, l4, l5, l6], dim=1)

        x = torch.cat([cls, loc], dim=2)

        return x, anchor

    def review(self, x, channel):
        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, (x.shape[0], -1, channel))
        return x

    def get_anchor(self, fh, fw, n_anchor, s1, s2):
        
        grid_interval_w = np.linspace(0.0, 1.0, fw+1)
        grid_interval_w = (grid_interval_w[:-1] + grid_interval_w[1:]) / 2.0
        grid_interval_h = np.linspace(0.0, 1.0, fh+1)
        grid_interval_h = (grid_interval_h[:-1] + grid_interval_h[1:]) / 2.0
        
        wgrid = np.tile(np.expand_dims(grid_interval_w, 0), (fh, 1))
        hgrid = np.tile(np.expand_dims(grid_interval_h, 1), (1, fw))
        
        cx = np.reshape(wgrid, (-1, 1))
        cy = np.reshape(hgrid, (-1, 1))
        
        na = fw * fh

        # 1: s1
        aw = s1
        ah = s1
        cw = np.tile(aw, (na, 1))
        ch = np.tile(ah, (na, 1))
        a1 = np.concatenate([cx, cy, cw, ch], axis=1)
        
        # 2: sqrt(s1, s2)
        aw = np.sqrt(s1 * s2)
        ah = np.sqrt(s1 * s2)
        cw = np.tile(aw, (na, 1))
        ch = np.tile(ah, (na, 1))
        a2 = np.concatenate([cx, cy, cw, ch], axis=1)
        
        # 3: 2x 1/2x
        # 1.4 1.61 1.82
        aw = s1 * 1.4
        ah = s1 / 1.4
        
        cw = np.tile(aw, (na, 1))
        ch = np.tile(ah, (na, 1))
        a3_1 = np.concatenate([cx, cy, cw, ch], axis=1)
        a3_2 = np.concatenate([cx, cy, ch, cw], axis=1)
        
        if n_anchor != 6:
            anchors = np.concatenate([a1, a2, a3_1, a3_2], axis=0)
            anchors = np.clip(anchors, 0.0, 1.0)
            return anchors
        else: 
            # 4: 3x 1/3x
            # 1.7 1.95 2.2
            aw = s1 * 1.7
            ah = s1 / 1.7
            
            cw = np.tile(aw, (na, 1))
            ch = np.tile(ah, (na, 1))
            a4_1 = np.concatenate([cx, cy, cw, ch], axis=1)
            a4_2 = np.concatenate([cx, cy, ch, cw], axis=1)
            anchors = np.concatenate([a1, a2, a3_1, a3_2, a4_1, a4_2], axis=0)
            anchors = np.clip(anchors, 0.0, 1.0)
            return anchors


class ssd_post(nn.Module):
    def __init__(self, ssd):
        super().__init__()

        self.ssd = ssd
        self.nc = ssd.nc
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x):
        x, anchor = self.ssd(x)
        x = self.postprocess(x, anchor)

        return x

    def postprocess(self, x, anchor):
        cls, d_x, d_y, d_w, d_h = torch.split(x[0], [self.nc, 1, 1, 1, 1], dim=1)
        a_x, a_y, a_w, a_h = torch.split(anchor, [1, 1, 1, 1], dim=1)
        
        cx = (d_x * a_w / 10.0) + a_x
        cy = (d_y * a_h / 10.0) + a_y
        w = torch.exp(d_w / 5.0) * a_w
        h = torch.exp(d_h / 5.0) * a_h

        x1 = cx - (w / 2.0)
        x2 = cx + (w / 2.0)
        y1 = cy - (h / 2.0)
        y2 = cy + (h / 2.0)
        
        cls = self.activation(cls)

        return torch.concat([cls, x1, y1, x2, y2], dim=1)

class ssd_manager():
    def __init__(self, config, ddp_rank):
        self.ddp_rank = ddp_rank
        
        self.phase = None # 'T' or 'V' for training and validate

        self.loss_class = AverageMeter()
        self.loss_location = AverageMeter()
        self.loss_all = AverageMeter()
        self.acc_fg = AverageMeter()
        self.acc_bg = AverageMeter()
        self.acc_location = AverageMeter()
        
        self.iter_start = time.time()
        self.iter_end = time.time()
        
        self.global_step = 0
        self.local_step = 0
        self.len_train = config['REPORT_STEP_TRAIN']
        self.len_valid = config['REPORT_STEP_VALID']

        if ddp_rank == 0 and config['REPORT_WANDB']:
            wandb.init(project='OD_SSD', entity='byunghyun', config=config)
        
    def report_wandb(self, rdT, rdV, step):
        if self.ddp_rank == 0:
            rdT.update(rdV)
            wandb.log(rdT, step=step)

    def report_model(self):

        pass
        
    def gather(self, l_all, l_cls, l_loc, a_fg, a_bg, a_loc):
        if self.ddp_rank == 0:
            self.loss_class.get_data(l_cls.detach().cpu().numpy())
            self.loss_location.get_data(l_loc.detach().cpu().numpy())
            self.loss_all.get_data(l_all.detach().cpu().numpy())
            self.acc_fg.get_data(a_fg)
            self.acc_bg.get_data(a_bg)
            self.acc_location.get_data(a_loc)
            
            self.global_step += 1
            self.local_step += 1
        
    def report(self):
        if self.ddp_rank == 0:
            report_dict = {
                self.phase+'Lall': self.loss_all.get_mean(),
                self.phase+'Lcls': self.loss_class.get_mean(), 
                self.phase+'Lloc': self.loss_location.get_mean(),
                self.phase+'Afg': self.acc_fg.get_mean(),
                self.phase+'Abg': self.acc_bg.get_mean(),
                self.phase+'Aloc': self.acc_location.get_mean(),
            }

            return report_dict
        
    def report_print(self):
        if self.ddp_rank == 0:
            self.iter_start = time.time()
            if self.phase == 'T':
                string = f'Training | step: {self.local_step}/{self.len_train} | '
            else:
                string = f'Validation | step: {self.local_step}/{self.len_valid} | '
            string += f'{1/(self.iter_start - self.iter_end):.2f} iter/S | '
            string += f'LAll: {self.loss_all.get_mean():.2f} | Lcls: {self.loss_class.get_mean():.2f} | LLoc: {self.loss_location.get_mean():.2f} | '
            string += f'Afg: {self.acc_fg.get_mean():.2f} | Abg: {self.acc_bg.get_mean():.2f} | ALoc: {self.acc_location.get_mean():.2f}'
            print(string, end='\r', flush=True)
            self.iter_end = time.time()
            
            sys.stdout.flush()

    def report_reset(self):
        if self.ddp_rank == 0:
            self.loss_class.reset()
            self.loss_location.reset()
            self.loss_all.reset()
            self.acc_fg.reset()
            self.acc_bg.reset()
            self.acc_location.reset()
            
            self.local_step = 0
            
            print("")


class ssd_augmentator(object):
    def __init__(self, config):
        input_size = config['INPUT_SIZE']

        self.transform_train = [
            augment_hsv(p=1.0),
            random_perspective(p=1.0),
            Normalize(), # 255로 나누기
            
            # RandomVFlip(p=0.5),
            # Oneof([
            #     RandomZoomIn(p=1.0, zoom_range=0.4),
            #     RandomZoomOut(p=1.0, zoom_range=0.4)
            # ], p=0.9),
            # RandomRotation(degree=5, p=0.2),

            # Oneof([
            #     brighter(brightness=0.25, p=1.0),
            #     darker(darkness=0.25, p=1.0)
            # ], p=0.2),
            
            DeNormalize(), # 255 곱하기
            Resize(input_size),
            ]

        self.transform_valid = [
            Normalize(),
            RandomVFlip(p=0.5),
            DeNormalize(),
            Resize(input_size),
            ]
            
    def __call__(self, img, labels, boxes, istrain):
        if istrain:
            for tform in self.transform_train:
                img, labels, boxes = tform(img, labels, boxes)
        else:
            for tform in self.transform_valid:
                img, labels, boxes = tform(img, labels, boxes)
        
        return img, labels, boxes

class ssd_preprocessor(object):
    def __init__(self, config, anchor):
        self.anchor = anchor.detach().cpu().numpy()
        self.nc = len(config['CLASS'])
    
    def __call__(self, label, box):
        gt = self.get_delta(label, box)
        return gt

    def set_nc(self, nc):
        self.nc = nc
        
    def get_delta(self, label, box):
        gtbox = np.array(box)
        cls_table = np.eye(self.nc)
        cls = []
        
        ngt = len(label)
        for i in range(ngt):
            cls.append(cls_table[label[i]])
        cls = np.array(cls)
        
        ious = self.jaccard_iou(self.anchor, gtbox)
        
        #assure at least 1 gt box
        assure_iou = np.equal(np.max(ious, axis=0), ious)
        assure_index = np.expand_dims(np.max(assure_iou, axis=1), axis=1)

        maxiou = np.expand_dims(np.max(ious, axis=1), axis=1)
        maxidx = np.argmax(ious, axis=1)
        pos = np.where(maxiou > 0.5, True, False)
        pos = np.logical_or(pos, assure_index)
        
        # class
        conf = cls[maxidx] * pos
        conf = np.concatenate([1-pos, conf], axis=1) # backgorund append
        
        # box regression
        target = gtbox[maxidx]
        
        delta_cx = np.expand_dims((target[:, 0] - self.anchor[:, 0]) / self.anchor[:, 2], axis=1) * 10.0
        delta_cy = np.expand_dims((target[:, 1] - self.anchor[:, 1]) / self.anchor[:, 3], axis=1) * 10.0
        delta_w = np.expand_dims(np.log(target[:, 2] / self.anchor[:, 2]), axis=1) * 5.0
        delta_h = np.expand_dims(np.log(target[:, 3] / self.anchor[:, 3]), axis=1) * 5.0
        
        delta = np.concatenate([delta_cx, delta_cy, delta_w, delta_h], axis=1)
        delta = delta * pos
        
        output = np.concatenate([conf, delta], axis=1)
        return output

    def jaccard_iou(self, box1, box2):
        # box1 : (cx, cy, w, h)
        # box2 : (cx, cy, w, h)
        nb1 = np.size(box1, 0)
        nb2 = np.size(box2, 0)

        box1 = np.expand_dims(box1, axis=1)
        box2 = np.expand_dims(box2, axis=0)
        
        box1 = np.tile(box1, (1, nb2, 1))
        box2 = np.tile(box2, (nb1, 1, 1))
        
        x1_box1 = box1[:, :, 0] - (box1[:, :, 2] / 2.0)
        x2_box1 = box1[:, :, 0] + (box1[:, :, 2] / 2.0)
        y1_box1 = box1[:, :, 1] - (box1[:, :, 3] / 2.0)
        y2_box1 = box1[:, :, 1] + (box1[:, :, 3] / 2.0)
        
        x1_box2 = box2[:, :, 0] - (box2[:, :, 2] / 2.0)
        x2_box2 = box2[:, :, 0] + (box2[:, :, 2] / 2.0)
        y1_box2 = box2[:, :, 1] - (box2[:, :, 3] / 2.0)
        y2_box2 = box2[:, :, 1] + (box2[:, :, 3] / 2.0)

        xmin = np.maximum(x1_box1, x1_box2)
        xmax = np.minimum(x2_box1, x2_box2)
        ymin = np.maximum(y1_box1, y1_box2)
        ymax = np.minimum(y2_box1, y2_box2)

        area1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        area2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
        
        common = np.maximum((xmax - xmin), 0.0) * np.maximum((ymax - ymin), 0.0)

        iou = common / (area1 + area2 - common)

        return iou