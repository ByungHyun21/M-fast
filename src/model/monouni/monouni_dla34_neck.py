import os, sys, yaml, time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision

from src.backbone.model.dla.dla34 import dla34

from src.layer import *

class monouni_dla34_neck(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.backbone = dla34(cfg)
        
        isdeformconv = False
        channels = [64, 128, 256, 512]

        self.x1_merge_1 = IDA_Up(channels[1], channels[0], channels[0], deformableconv=isdeformconv)
        self.x1_merge_2 = IDA_Up(channels[1], channels[0], channels[0], deformableconv=isdeformconv)
        self.x1_merge_3 = IDA_Up(channels[1], channels[0], channels[0], deformableconv=isdeformconv)

        self.x2_merge_1 = IDA_Up(channels[2], channels[1], channels[1], deformableconv=isdeformconv)
        self.x2_merge_2 = IDA_Up(channels[2], channels[1], channels[1], deformableconv=isdeformconv)

        self.x3_merge_1 = IDA_Up(channels[3], channels[2], channels[2], deformableconv=isdeformconv)

        self.out_merge1 = IDA_Up(channels[1], channels[0], channels[0], scale=1, deformableconv=isdeformconv)
        self.out_merge2 = IDA_Up(channels[2], channels[0], channels[0], scale=2, deformableconv=isdeformconv)
        self.out_merge3 = IDA_Up(channels[3], channels[0], channels[0], scale=4, deformableconv=isdeformconv)
        
    def forward(self, x):
        x = self.backbone(x)
        
        # Neck
        _, _, x1, x2, x3, x4 = x
        # (in)
        # x1 ------ x1 ------ x1 ------ x1
        # |    /        /        /       |
        # x2 ------ x2 ------ x2 ------  x
        # |    /        /                |
        # x3 ------ x3 ----------------  x
        # |    /                         |
        # x4 --------------------------  x(out)

        x1 = self.x1_merge_1(x2, x1)
        x2 = self.x2_merge_1(x3, x2)
        x3 = self.x3_merge_1(x4, x3)

        x1 = self.x1_merge_2(x2, x1)
        x2 = self.x2_merge_2(x3, x2)
        
        x1 = self.x1_merge_3(x2, x1)

        x = self.out_merge1(x2, x1)
        x = self.out_merge2(x3, x)
        x = self.out_merge3(x4, x)

        return x
    
    def load_pretrained_backbone(self, path):
        # self.backbone.load_state_dict(torch.load(path, map_location='cpu'))
        pass
    
    def load_pretrained_model(self, path):
        # self.load_state_dict(torch.load(path, map_location='cpu'))
        pass
        