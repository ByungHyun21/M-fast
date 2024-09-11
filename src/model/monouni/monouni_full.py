import torch
import torch.nn as nn

import numpy as np

from src.layer import *

class monouni_full(nn.Module):
    def __init__(self, cfg, backbone_neck):
        super().__init__()
        
        self.model = monouni_head(cfg, backbone_neck)
        
        self.interval_min = [0, 30, 60, 90, 120]
        self.interval_max = [40, 70, 100, 130, 180]
        
        pass
    
    def forward(self, x):
        x = self.model(x)
        
        
        return x
    
    def postprocess(self, x):
        
        pass
        
class monouni_head(nn.Module):
    def __init__(self, cfg, backbone_neck):
        super().__init__()
        
        self.backbone_neck = backbone_neck
        
        n_class = len(cfg['classes'])
        
        self.heatmap_0 = Conv2D(64, 256, k=3, s=1, p=1, bias=True, bn=None, act='relu')
        self.heatmap_1 = Conv2D(256, n_class, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.offset2d_0 = Conv2D(64, 256, k=3, s=1, p=1, bias=True, bn=None, act='relu')
        self.offset2d_1 = Conv2D(256, 2, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.size2d_0 = Conv2D(64, 256, k=3, s=1, p=1, bias=True, bn=None, act='relu')
        self.size2d_1 = Conv2D(256, 2, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.offset3d_0 = Conv2D(128+n_class+2, 256, k=3, s=1, p=1, bias=True, bn=['bn'], act='relu')
        self.offset3d_1 = nn.AdaptiveAvgPool2d(1)
        self.offset3d_2 = Conv2D(256, 2, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.size3d_0 = Conv2D(128+n_class+2, 256, k=3, s=1, p=1, bias=True, bn=['bn'], act='relu')
        self.size3d_1 = nn.AdaptiveAvgPool2d(1)
        self.size3d_2 = Conv2D(256, 3, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.heading_0 = Conv2D(128+n_class+2, 256, k=3, s=1, p=1, bias=True, bn=['bn'], act='relu')
        self.heading_1 = nn.AdaptiveAvgPool2d(1)
        self.heading_2 = Conv2D(256, 24, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.vis_depth_0 = Conv2D(128+n_class+2, 256, k=3, s=1, p=1, bias=True, bn=None, act='leakyrelu')
        self.vis_depth_1 = Conv2D(256, 5, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.att_depth_0 = Conv2D(128+n_class+2, 256, k=3, s=1, p=1, bias=True, bn=None, act='leakyrelu')
        self.att_depth_1 = Conv2D(256, 5, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.vis_depth_uncertainty_0 = Conv2D(128+n_class+2, 256, k=3, s=1, p=1, bias=True, bn=None, act='leakyrelu')
        self.vis_depth_uncertainty_1 = Conv2D(256, 5, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.att_depth_uncertainty_0 = Conv2D(128+n_class+2, 256, k=3, s=1, p=1, bias=True, bn=None, act='leakyrelu')
        self.att_depth_uncertainty_1 = Conv2D(256, 5, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        self.depth_bin_0 = Conv2D(128+n_class+2, 256, k=3, s=1, p=1, bias=True, bn=None, act='leakyrelu')
        self.depth_bin_1 = nn.AdaptiveAvgPool2d(1)
        self.depth_bin_2 = Conv2D(256, 10, k=1, s=1, p=0, bias=True, bn=None, act=None)
        
        pass
    
    def forward(self, x, mode='train'):
        x = self.backbone_neck(x)
        
        heatmap = self.heatmap_0(x)
        heatmap = self.heatmap_1(heatmap)
        
        offset_2d = self.offset2d_0(x)
        offset_2d = self.offset2d_1(offset_2d)
        
        size2d = self.size2d_0(x)
        size2d = self.size2d_1(size2d)
        
        result = {}
        result['heatmap'] = heatmap
        result['offset_2d'] = offset_2d
        result['size2d'] = size2d
        
        # TODO: Postprocess 
        
        return x