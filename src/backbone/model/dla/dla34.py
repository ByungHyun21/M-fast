import os, sys, yaml, time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision

from src.layer import *

class dla34(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        use_bias = False
    
        self.base = Conv2D(3, 16, k=7, s=1, p=3, bn=['GN', 16], bias=use_bias, act='relu')
        self.conv2d_1 = Conv2D(16, 16, bn=['GN', 16], bias=use_bias, act='relu')
        self.conv2d_2 = Conv2D(16, 32, s=2, bn=['GN', 32], bias=use_bias, act='relu')
        
        self.block1_dla1 = DLA_Module(32, 64, 0, k=3, s=2, p=1, bn=['GN', 32], bias=use_bias, act='relu')

        self.block2_dla1 = DLA_Module(64, 128, 0, k=3, s=2, p=1, bn=['GN', 64], bias=use_bias, act='relu')
        self.block2_dla2 = DLA_Module(128, 128, 64, k=3, s=1, p=1, bn=['GN', 128], bias=use_bias, act='relu')
        self.block2_downsample = MaxPool2D(k=2, s=2, p=0, ceil_mode=False)

        self.block3_dla1 = DLA_Module(128, 256, 0, k=3, s=2, p=1, bn=['GN', 128], bias=use_bias, act='relu')
        self.block3_dla2 = DLA_Module(256, 256, 128, k=3, s=1, p=1, bn=['GN', 256], bias=use_bias, act='relu')
        self.block3_downsample = MaxPool2D(k=2, s=2, p=0, ceil_mode=False)

        self.block4_dla1 = DLA_Module(256, 512, 256, k=3, s=2, p=1, bn=['GN', 256], bias=use_bias, act='relu')
        self.block4_downsample = MaxPool2D(k=2, s=2, p=0, ceil_mode=False)
        
        
    def forward(self, x):
        x = self.base(x)
        x1 = self.conv2d_1(x)
        x2 = self.conv2d_2(x1)
        x3 = self.block1_dla1(x2)

        x4_1 = self.block2_dla1(x3)
        x3_downsample = self.block2_downsample(x3)
        x4 = self.block2_dla2(x4_1, x3_downsample)

        x5_1 = self.block3_dla1(x4)
        x4_downsample = self.block3_downsample(x4)
        x5 = self.block3_dla2(x5_1, x4_downsample)

        x5_downsample = self.block4_downsample(x5)
        x6 = self.block4_dla1(x5, x5_downsample)

        return (x1, x2, x3, x4, x5, x6)