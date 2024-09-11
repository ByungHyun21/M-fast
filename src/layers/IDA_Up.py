from typing import Union, Optional
from .utils import *

import torch
import torch.nn as nn

from src.layers.Conv2D import Conv2D
from src.layers.DeConv2D import DeConv2D

class IDA_Up(nn.Module):
    def __init__(self, 
                 cin_low, 
                 cin_high, 
                 cout, 
                 scale:int=1, 
                 deformableconv:bool=False):
        # cin_low : 낮은 level의 input channel
        # cin_high : 높은 level의 input channel
        super().__init__()
        if deformableconv:
            #TODO: Deformable Convolution 구현 필요
            pass
        else:
            self.conv1 = Conv2D(cin_low, cout, bn=['GN', 32], bias=False, act='relu')
            self.conv2 = Conv2D(cin_high, cout, bn=['GN', 32], bias=False, act='relu')
        
        self.deconv = DeConv2D(cout, cout, k=4*scale, s=2*scale, p=1*scale, g=cout, bn=None, bias=False, act=None)

    def forward(self, x_low, x_high):
        x_low = self.deconv(self.conv1(x_low))
        x = self.conv2(x_low + x_high)
        return x