import os, sys, yaml, time, wandb
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision


from model.layers import *

class ssd_vgg16(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.ddp_rank = ddp_rank
        self.model_name = config['METHOD'] + '_' + config['TYPE']
        self.nc = len(config['CLASS']) + 5 # number of class, 5 is for background(1), regression(4)
        self.device = config['DEVICE']
        self.model_class = config['CLASS']
        
        anchor_n = config['ANCHOR_N']
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        self.backbone = nn.Sequential(
            # Conv 1
            Conv2d(3, 64, act='relu'),           # 300x300x3 -> 300x300x64
            Conv2d(64, 64, act='relu'),          # 300x300x64 -> 300x300x64
            MaxPool2d(k=2, s=2, p=0),            # 300x300x64 -> 150x150x64
            
            # Conv 2
            Conv2d(64, 128, act='relu'),         # 150x150x64 -> 150x150x128
            Conv2d(128, 128, act='relu'),        # 150x150x128 -> 150x150x128
            MaxPool2d(k=2, s=2, p=0),            # 150x150x128 -> 75x75x128
            
            # Conv 3
            Conv2d(128, 256, act='relu'),        # 75x75x128 -> 75x75x256
            Conv2d(256, 256, act='relu'),        # 75x75x256 -> 75x75x256
            Conv2d(256, 256, act='relu'),        # 75x75x256 -> 75x75x256
            MaxPool2d(k=2, s=2, p=0),            # 75x75x256 -> 38x38x256
        )
        
        self.extra_1 = nn.Sequential(
            Conv2d(256, 512, act='relu'),        # 38x38x256 -> 38x38x512
            Conv2d(512, 512, act='relu'),        # 38x38x512 -> 38x38x512
            Conv2d(512, 512, act='relu'),        # 38x38x512 -> 38x38x512
        )
        self.extra_2 = nn.Sequential(
            MaxPool2d(k=2, s=2, p=0),            # 38x38x512 -> 19x19x512
            Conv2d(512, 512, act='relu'),        # 19x19x512 -> 19x19x512
            Conv2d(512, 512, act='relu'),        # 19x19x512 -> 19x19x512
            Conv2d(512, 512, act='relu'),        # 19x19x512 -> 19x19x512
            MaxPool2d(k=3, s=1, p=1),            # 19x19x512 -> 19x19x512
            Conv2d(512, 1024, act='relu'),       # 19x19x512 -> 19x19x1024
            Conv2d(1024, 1024, act='relu'),      # 19x19x1024 -> 19x19x1024
        )
        self.extra_3 = nn.Sequential(
            Conv2d(1024, 256, k=1, act='relu'),      # 19x19x1024 -> 19x19x256
            Conv2d(256, 512, k=3, s=2, act='relu'),  # 19x19x256 -> 10x10x512
        )
        self.extra_4 = nn.Sequential(
            Conv2d(512, 128, k=1, act='relu'),       # 10x10x512 -> 10x10x128
            Conv2d(128, 256, k=3, s=2, act='relu'),  # 10x10x128 -> 5x5x256
        )
        self.extra_5 = nn.Sequential(
            Conv2d(256, 128, k=1, act='relu'),       # 5x5x256 -> 5x5x128
            Conv2d(128, 256, k=3, p=0, act='relu'),  # 5x5x128 -> 3x3x256
        )
        self.extra_6 = nn.Sequential(
            Conv2d(256, 128, k=1, act='relu'),       # 3x3x256 -> 3x3x128
            Conv2d(128, 256, k=3, p=0, act='relu'),  # 3x3x128 -> 1x1x256
        )
        
        self.head_1 = Conv2d(512, anchor_n[0]*self.nc, bn=False, act=None)   # 19x19x576 -> 19x19x(anchors*nc)
        self.head_2 = Conv2d(1024, anchor_n[1]*self.nc, bn=False, act=None)   # 10x10x160 -> 10x10x(anchors*nc)
        self.head_3 = Conv2d(512, anchor_n[2]*self.nc, bn=False, act=None)   # 5x5x256 -> 5x5x(anchors*nc)
        self.head_4 = Conv2d(256, anchor_n[3]*self.nc, bn=False, act=None)   # 3x3x256 -> 3x3x(anchors*nc)
        self.head_5 = Conv2d(256, anchor_n[4]*self.nc, bn=False, act=None)   # 2x2x256 -> 2x2x(anchors*nc)
        self.head_6 = Conv2d(256, anchor_n[5]*self.nc, bn=False, act=None)   # 1x1x128 -> 1x1x(anchors*nc)
            
        self.rescale_factor = nn.Parameter(torch.FloatTensor(1, 512, 1, 1), requires_grad=True)
        nn.init.constant_(self.rescale_factor, 20)

        self.backbone_weight = config['BACKBONE_WEIGHT']
        self.pretrained_weight = config['PRETRAINED_WEIGHT']

        self.load_torchvision_weights()
        self.load_pretrained_weights()

    def forward(self, x):
        # x = (x / 128.0) - 1.0
        
        x = x / 255.0
        x = (x - self.mean) / self.std
        
        x = self.backbone(x)
        
        x1 = self.extra_1(x)
        x2 = self.extra_2(x1)
        x3 = self.extra_3(x2)
        x4 = self.extra_4(x3)
        x5 = self.extra_5(x4)
        x6 = self.extra_6(x5)
        
        norm = x1.pow(2).sum(dim=1, keepdim=True).sqrt()
        x1 = x1 / norm
        x1 = x1 * self.rescale_factor

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
        
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        
        return x
    
    def review(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(x.shape[0], -1, self.nc)
        return x
    
    def load_torchvision_weights(self):
        if self.backbone_weight == 'torchvision':
            state_dict = self.state_dict()
            param_names = list(state_dict.keys())

            pretrained_dict = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT).state_dict()
            pretrained_names = list(pretrained_dict.keys())

            for i, name in enumerate(param_names[1:]): # exclude [0]는 rescale_factor임. weight loading에서 제외
                if 'backbone' in name or 'extra_1' in name:
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