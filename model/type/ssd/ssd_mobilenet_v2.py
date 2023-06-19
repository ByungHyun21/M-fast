import os, sys, yaml, time, wandb
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision

from model.layers import *

class ssd_mobilenet_v2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg['exp_name']
        self.nc = cfg['network']['num_classes'] + 1 # number of class, 1 is for background(1)
        self.device = cfg['device']
        self.model_class = cfg['network']['classes']
        
        anchor_n = cfg['anchor']['num_anchor']

        self.mean = torch.tensor(cfg['network']['mean']).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(cfg['network']['std']).view(1, 3, 1, 1).to(self.device)

        self.backbone = nn.Sequential(
            Conv2d(3, 32, s=2, bias=False, act='relu6'),     # 300x300x3 -> 150x150x32
            GroupConv2d(32, 32, bias=False, act='relu6'),    # 150x150x32 -> 150x150x32
            Conv2d(32, 16, k=1, bias=False, act=None),       # 150x150x32 -> 150x150x16
            
            MobileNet_V2_Conv2d(16, 24, s=2, bias=False),    # 150x150x16 -> 75x75x24
            MobileNet_V2_Conv2d(24, 24, bias=False),         # 75x75x24 -> 75x75x24
            MobileNet_V2_Conv2d(24, 32, s=2, bias=False),    # 75x75x24 -> 38x38x32
            MobileNet_V2_Conv2d(32, 32, bias=False),         # 38x38x32 -> 38x38x32
            MobileNet_V2_Conv2d(32, 32, bias=False),         # 38x38x32 -> 38x38x32
            MobileNet_V2_Conv2d(32, 64, s=2, bias=False),    # 38x38x32 -> 19x19x64
            MobileNet_V2_Conv2d(64, 64, bias=False),         # 19x19x64 -> 19x19x64
            MobileNet_V2_Conv2d(64, 64, bias=False),         # 19x19x64 -> 19x19x64
            MobileNet_V2_Conv2d(64, 64, bias=False),         # 19x19x64 -> 19x19x64
            MobileNet_V2_Conv2d(64, 96, bias=False),         # 19x19x64 -> 19x19x96
            MobileNet_V2_Conv2d(96, 96, bias=False),         # 19x19x96 -> 19x19x96
            MobileNet_V2_Conv2d(96, 96, bias=False),         # 19x19x96 -> 19x19x96
        )
        
        self.extra_1 = nn.Sequential(
            Conv2d(96, 576, k=1, bias=False, act='relu6'),   # 19x19x96 -> 19x19x576
        )
        self.extra_2 = nn.Sequential(
            GroupConv2d(576, 576, s=2, bias=False, act='relu6'), # 19x19x576 -> 10x10x576
            Conv2d(576, 160, k=1, bias=False, act=None),     # 10x10x576 -> 10x10x160
            MobileNet_V2_Conv2d(160, 160, bias=False),       # 10x10x160 -> 10x10x160
            MobileNet_V2_Conv2d(160, 160, bias=False),       # 10x10x160 -> 10x10x160
            MobileNet_V2_Conv2d(160, 320, bias=False),       # 10x10x160 -> 10x10x320
            Conv2d(320, 1280, k=1, bias=False, act='relu6'), # 10x10x320 -> 10x10x1280
        )
        self.extra_3 = nn.Sequential(
            Conv2d(1280, 256, k=1, act='relu6'), # 10x10x1280 -> 10x10x256
            GroupConv2d(256, 256, s=2, act='relu6'), # 10x10x256 -> 5x5x256
            Conv2d(256, 512, k=1, act='relu6'),  # 5x5x256 -> 5x5x512
        )
        self.extra_4 = nn.Sequential(
            Conv2d(512, 128, k=1, act='relu6'),  # 5x5x512 -> 5x5x128
            GroupConv2d(128, 128, s=2, act='relu6'), # 5x5x128 -> 3x3x128
            Conv2d(128, 256, k=1, act='relu6'),  # 3x3x128 -> 3x3x256
        )
        self.extra_5 = nn.Sequential(
            Conv2d(256, 128, k=1, act='relu6'),  # 3x3x256 -> 3x3x128
            GroupConv2d(128, 128, s=2, act='relu6'), # 3x3x128 -> 2x2x128
            Conv2d(128, 256, k=1, act='relu6'),  # 2x2x128 -> 2x2x256
        )
        self.extra_6 = nn.Sequential(
            Conv2d(256, 64, k=1, act='relu6'),   # 2x2x256 -> 2x2x64
            GroupConv2d(64, 64, s=2, act='relu6'),  # 2x2x64 -> 1x1x64
            Conv2d(64, 128, k=1, act='relu6'),   # 1x1x64 -> 1x1x128
        )
        
        # self.head_1 = Conv2d(576, anchor_n[0]*self.nc, bn=False, act=None)   # 19x19x576 -> 19x19x(anchors*nc)
        # self.head_2 = Conv2d(1280, anchor_n[1]*self.nc, bn=False, act=None)   # 10x10x160 -> 10x10x(anchors*nc)
        # self.head_3 = Conv2d(512, anchor_n[2]*self.nc, bn=False, act=None)   # 5x5x256 -> 5x5x(anchors*nc)
        # self.head_4 = Conv2d(256, anchor_n[3]*self.nc, bn=False, act=None)   # 3x3x256 -> 3x3x(anchors*nc)
        # self.head_5 = Conv2d(256, anchor_n[4]*self.nc, bn=False, act=None)   # 2x2x256 -> 2x2x(anchors*nc)
        # self.head_6 = Conv2d(128, anchor_n[5]*self.nc, bn=False, act=None)   # 1x1x128 -> 1x1x(anchors*nc)
        
        self.head_cls1 = Conv2d(576, anchor_n[0]*self.nc, bn=False, act=None)   # 19x19x576 -> 19x19x(anchors*nc)
        self.head_cls2 = Conv2d(1280, anchor_n[1]*self.nc, bn=False, act=None)   # 10x10x160 -> 10x10x(anchors*nc)
        self.head_cls3 = Conv2d(512, anchor_n[2]*self.nc, bn=False, act=None)   # 5x5x256 -> 5x5x(anchors*nc)
        self.head_cls4 = Conv2d(256, anchor_n[3]*self.nc, bn=False, act=None)   # 3x3x256 -> 3x3x(anchors*nc)
        self.head_cls5 = Conv2d(256, anchor_n[4]*self.nc, bn=False, act=None)   # 2x2x256 -> 2x2x(anchors*nc)
        self.head_cls6 = Conv2d(128, anchor_n[5]*self.nc, bn=False, act=None)   # 1x1x128 -> 1x1x(anchors*nc)

        self.head_loc1 = Conv2d(576, anchor_n[0]*4, bn=False, act=None)   # 19x19x576 -> 19x19x(anchors*4)
        self.head_loc2 = Conv2d(1280, anchor_n[1]*4, bn=False, act=None)   # 10x10x160 -> 10x10x(anchors*4)
        self.head_loc3 = Conv2d(512, anchor_n[2]*4, bn=False, act=None)   # 5x5x256 -> 5x5x(anchors*4)
        self.head_loc4 = Conv2d(256, anchor_n[3]*4, bn=False, act=None)   # 3x3x256 -> 3x3x(anchors*4)
        self.head_loc5 = Conv2d(256, anchor_n[4]*4, bn=False, act=None)   # 2x2x256 -> 2x2x(anchors*4)
        self.head_loc6 = Conv2d(128, anchor_n[5]*4, bn=False, act=None)   # 1x1x128 -> 1x1x(anchors*4)
        
        self.sigmoid = nn.Sigmoid()

        if cfg['network']['init_weight'] == 'torchvision':
            self.load_torchvision_weights()
        elif cfg['network']['init_weight'] is not None:
            self.load_pretrained_weights()

    def forward(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        
        x = self.backbone(x)
        
        x1 = self.extra_1(x)
        x2 = self.extra_2(x1)
        x3 = self.extra_3(x2)
        x4 = self.extra_4(x3)
        x5 = self.extra_5(x4)
        x6 = self.extra_6(x5)
        
        # x1 = self.head_1(x1)
        # x2 = self.head_2(x2)
        # x3 = self.head_3(x3)
        # x4 = self.head_4(x4)
        # x5 = self.head_5(x5)
        # x6 = self.head_6(x6)
        
        # x1 = self.review(x1)
        # x2 = self.review(x2)
        # x3 = self.review(x3)
        # x4 = self.review(x4)
        # x5 = self.review(x5)
        # x6 = self.review(x6)
        
        # x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1).contiguous()
        
        cls_x1 = self.head_cls1(x1)
        cls_x2 = self.head_cls2(x2)
        cls_x3 = self.head_cls3(x3)
        cls_x4 = self.head_cls4(x4)
        cls_x5 = self.head_cls5(x5)
        cls_x6 = self.head_cls6(x6)
        
        cls_x1 = self.review(cls_x1, self.nc)
        cls_x2 = self.review(cls_x2, self.nc)
        cls_x3 = self.review(cls_x3, self.nc)
        cls_x4 = self.review(cls_x4, self.nc)
        cls_x5 = self.review(cls_x5, self.nc)
        cls_x6 = self.review(cls_x6, self.nc)
        
        loc_x1 = self.head_loc1(x1)
        loc_x2 = self.head_loc2(x2)
        loc_x3 = self.head_loc3(x3)
        loc_x4 = self.head_loc4(x4)
        loc_x5 = self.head_loc5(x5)
        loc_x6 = self.head_loc6(x6)
        
        loc_x1 = self.review(loc_x1, 4)
        loc_x2 = self.review(loc_x2, 4)
        loc_x3 = self.review(loc_x3, 4)
        loc_x4 = self.review(loc_x4, 4)
        loc_x5 = self.review(loc_x5, 4)
        loc_x6 = self.review(loc_x6, 4)
        
        classification = torch.cat([cls_x1, cls_x2, cls_x3, cls_x4, cls_x5, cls_x6], dim=1).contiguous()
        classification = self.sigmoid(classification)
        
        location = torch.cat([loc_x1, loc_x2, loc_x3, loc_x4, loc_x5, loc_x6], dim=1).contiguous()
        
        output = torch.cat([classification, location], dim=2).contiguous()
        return output
    
    def review(self, x, last_channel):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(x.shape[0], -1, last_channel)
        return x
    
    def load_torchvision_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_dict = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).state_dict()
        pretrained_names = list(pretrained_dict.keys())

        for i, name in enumerate(param_names):
            if 'backbone' in name or 'extra_1' in name or 'extra_2' in name:
                # print(name, '<', pretrained_names[i])
                state_dict[name] = pretrained_dict[pretrained_names[i]]

                if pretrained_names[i][-5:] != name[-5:] or \
                    pretrained_dict[pretrained_names[i]].shape != state_dict[name].shape:
                    # print(state_dict[name].shape)
                    # print(pretrained_dict[pretrained_names[i]].shape)
                    assert False, 'Pretrained model과 our model의 shape이 일치하지 않음'

        self.load_state_dict(state_dict)

    def load_pretrained_weights(self):
        

        pass
    



