from typing import Union, Optional
from .utils import *

import torch
import torch.nn as nn
import torch.nn.init as init

from src.layers.Conv2D import Conv2D
from src.layers.GroupConv2D import GroupConv2D

class MobileNet_V2_Conv2D(nn.Module):
    """
    cin : input channel
    cout : output channel
    s : stride
    bias : bias
    """
    def __init__(self, 
                 cin:int, 
                 cout:int, 
                 s:int=1, 
                 bn:Union[list, None]=['bn'],
                 bias:bool=True):
        super().__init__()
        self.residual = False
        if cin == cout and s == 1:
            # cin == cout : residual connection (same channel)
            # s == 1 : residual connection (no stride)
            self.residual = True

        self.Conv2d_1 = Conv2D(cin, cin*6, 1, bn=bn, bias=bias, act='relu6')
        self.GroupConv2d = GroupConv2D(cin*6, cin*6, s=s, bn=bn, bias=bias, act='relu6')
        self.Conv2d_2 = Conv2D(cin*6, cout, 1, bn=bn, bias=bias, act=None)

    def forward(self, x_in):
        x = self.Conv2d_1(x_in)
        x = self.GroupConv2d(x)
        x = self.Conv2d_2(x)

        if self.residual:
            x = x + x_in

        return x