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
"""
Model List

- VGG16-SSD
- MobileNet V1 SSD
- MobileNet V2 SSD
- MobileNet V3 SSD
"""

class ssd(nn.Module):
    def __init__(self, config, ddp_rank:int):
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

