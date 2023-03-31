import os, sys, yaml, time, wandb
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from model.layers import *

class ssd_mobilenet_v2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.ddp_rank = ddp_rank
        self.model_name = config['METHOD'] + '_' + config['TYPE']
        self.nc = len(config['CLASS']) + 5 # number of class, 5 is for background(1), regression(4)
        self.device = config['DEVICE']
        self.model_class = config['CLASS']
        
        bias = config['BIAS']
        anchor_n = config['ANCHOR_N']
        
        # self.mean = torch.reshape(torch.tensor(config['MEAN'], device=self.ddp_rank), (1, -1, 1, 1))
        # self.std = torch.reshape(torch.tensor(config['STD'], device=self.ddp_rank), (1, -1, 1, 1))
        
        self.backbone = nn.Sequential(
            Conv2d(3, 32, s=2, bias=bias, act='relu6'),     # 300x300x3 -> 150x150x32
            GroupConv2d(32, 32, bias=bias, act='relu6'),    # 150x150x32 -> 150x150x32
            Conv2d(32, 16, k=1, bias=bias, act=None),       # 150x150x32 -> 150x150x16
            
            MobileNet_V2_Conv2d(16, 24, s=2, bias=bias),    # 150x150x16 -> 75x75x24
            MobileNet_V2_Conv2d(24, 24, bias=bias),         # 75x75x24 -> 75x75x24
            MobileNet_V2_Conv2d(24, 32, s=2, bias=bias),    # 75x75x24 -> 38x38x32
            MobileNet_V2_Conv2d(32, 32, bias=bias),         # 38x38x32 -> 38x38x32
            MobileNet_V2_Conv2d(32, 32, bias=bias),         # 38x38x32 -> 38x38x32
            MobileNet_V2_Conv2d(32, 64, s=2, bias=bias),    # 38x38x32 -> 19x19x64
            MobileNet_V2_Conv2d(64, 64, bias=bias),         # 19x19x64 -> 19x19x64
            MobileNet_V2_Conv2d(64, 64, bias=bias),         # 19x19x64 -> 19x19x64
            MobileNet_V2_Conv2d(64, 64, bias=bias),         # 19x19x64 -> 19x19x64
            MobileNet_V2_Conv2d(64, 96, bias=bias),         # 19x19x64 -> 19x19x96
            MobileNet_V2_Conv2d(96, 96, bias=bias),         # 19x19x96 -> 19x19x96
            MobileNet_V2_Conv2d(96, 96, bias=bias),         # 19x19x96 -> 19x19x96
        )
        
        self.extra_1 = nn.Sequential(
            Conv2d(96, 576, k=1, bias=bias, act='relu6'),   # 19x19x96 -> 19x19x576
        )
        self.extra_2 = nn.Sequential(
            GroupConv2d(576, 576, s=2, bias=bias, act='relu6'), # 19x19x576 -> 10x10x576
            Conv2d(576, 160, k=1, bias=bias, act=None),     # 10x10x576 -> 10x10x160
            MobileNet_V2_Conv2d(160, 160, bias=bias),       # 10x10x160 -> 10x10x160
            MobileNet_V2_Conv2d(160, 160, bias=bias),       # 10x10x160 -> 10x10x160
            MobileNet_V2_Conv2d(160, 320, bias=bias),       # 10x10x160 -> 10x10x320
            Conv2d(320, 1280, k=1, bias=bias, act='relu6'), # 10x10x320 -> 10x10x1280
        )
        self.extra_3 = nn.Sequential(
            Conv2d(1280, 256, k=1, bias=bias, act='relu6'), # 10x10x1280 -> 10x10x256
            GroupConv2d(256, 256, s=2, bias=bias, act='relu6'), # 10x10x256 -> 5x5x256
            Conv2d(256, 512, k=1, bias=bias, act='relu6'),  # 5x5x256 -> 5x5x512
        )
        self.extra_4 = nn.Sequential(
            Conv2d(512, 128, k=1, bias=bias, act='relu6'),  # 5x5x512 -> 5x5x128
            GroupConv2d(128, 128, s=2, bias=bias, act='relu6'), # 5x5x128 -> 3x3x128
            Conv2d(128, 256, k=1, bias=bias, act='relu6'),  # 3x3x128 -> 3x3x256
        )
        self.extra_5 = nn.Sequential(
            Conv2d(256, 128, k=1, bias=bias, act='relu6'),  # 3x3x256 -> 3x3x128
            GroupConv2d(128, 128, s=2, bias=bias, act='relu6'), # 3x3x128 -> 2x2x128
            Conv2d(128, 256, k=1, bias=bias, act='relu6'),  # 2x2x128 -> 2x2x256
        )
        self.extra_6 = nn.Sequential(
            Conv2d(256, 64, k=1, bias=bias, act='relu6'),   # 2x2x256 -> 2x2x64
            GroupConv2d(64, 64, s=2, bias=bias, act='relu6'),  # 2x2x64 -> 1x1x64
            Conv2d(64, 128, k=1, bias=bias, act='relu6'),   # 1x1x64 -> 1x1x128
        )
        
        self.head_1 = Conv2d(576, anchor_n[0]*self.nc, bn=False, bias=True, act=None)   # 19x19x576 -> 19x19x(anchors*nc)
        self.head_2 = Conv2d(1280, anchor_n[1]*self.nc, bn=False, bias=True, act=None)   # 10x10x160 -> 10x10x(anchors*nc)
        self.head_3 = Conv2d(512, anchor_n[2]*self.nc, bn=False, bias=True, act=None)   # 5x5x256 -> 5x5x(anchors*nc)
        self.head_4 = Conv2d(256, anchor_n[3]*self.nc, bn=False, bias=True, act=None)   # 3x3x256 -> 3x3x(anchors*nc)
        self.head_5 = Conv2d(256, anchor_n[4]*self.nc, bn=False, bias=True, act=None)   # 2x2x256 -> 2x2x(anchors*nc)
        self.head_6 = Conv2d(128, anchor_n[5]*self.nc, bn=False, bias=True, act=None)   # 1x1x128 -> 1x1x(anchors*nc)
            

        self.backbone.apply(weights_init)
        self.extra_1.apply(weights_init)
        self.extra_2.apply(weights_init)
        self.extra_3.apply(weights_init)
        self.extra_4.apply(weights_init)
        self.extra_5.apply(weights_init)
        self.extra_6.apply(weights_init)
        self.head_1.apply(weights_init)
        self.head_2.apply(weights_init)
        self.head_3.apply(weights_init)
        self.head_4.apply(weights_init)
        self.head_5.apply(weights_init)
        self.head_6.apply(weights_init)

    def forward(self, x):
        x = (x / 128.0) - 1.0
        # x = (x - self.mean) / self.std
        
        x = self.backbone(x)
        
        x1 = self.extra_1(x)
        x2 = self.extra_2(x1)
        x3 = self.extra_3(x2)
        x4 = self.extra_4(x3)
        x5 = self.extra_5(x4)
        x6 = self.extra_6(x5)
        
        x1 = self.head_1(x1)
        x2 = self.head_2(x2)
        x3 = self.head_3(x3)
        x4 = self.head_4(x4)
        x5 = self.head_5(x5)
        x6 = self.head_6(x6)
        
        x1 = self.review(x1)
        x2 = self.review(x2)
        x3 = self.review(x3)
        x4 = self.review(x4)
        x5 = self.review(x5)
        x6 = self.review(x6)
        
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1).contiguous()
        
        return x
    
    def review(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(x.shape[0], -1, self.nc).contiguous()
        return x
    
    



