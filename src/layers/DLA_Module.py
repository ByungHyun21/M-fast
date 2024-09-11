from typing import Union, Optional
from .utils import *

import torch
import torch.nn as nn

from src.layers.Conv2D import Conv2D
from src.layers.ResNet_Conv2D import ResNet_Conv2D
from src.layers.MaxPool2D import MaxPool2D

class DLA_Module(nn.Module):
    """
    Deep Layer Aggregation Module
    
    cin : input channel
    cout : output channel
    c_children : channel of children
    k : kernel size
    s : stride (important)
    p : padding
    d : dilation
    bias : bias
    pm : padding mode
    act : activation function
    """
    def __init__(self, 
                 cin:int, 
                 cout:int, 
                 c_children:int, 
                 k:int=3, 
                 s:int=1, 
                 p:Union[int, None]=None, 
                 d:int=1, 
                 bn:Union[list, None]=['bn'],
                 bias:bool=True, 
                 pm:str='zeros', 
                 act:Union[str, None]='relu'):
        super().__init__()
        
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        
        c_root_in = 2*cout + c_children

        self.tree1 = ResNet_Conv2D(cin, cout, k, s, p, d, bn, bias, pm, act)
        self.tree2 = ResNet_Conv2D(cout, cout, k, 1, p, d, bn, bias, pm, act)
        self.root = Conv2D(c_root_in, cout, 1, 1, 0, 1, bn, bias, pm, act)
        self.downsample = MaxPool2D(k=2, s=2, p=0, ceil_mode=False) if s > 1 else None
        self.project = Conv2D(cin, cout, 1, 1, 0, 1, bn, bias, pm, None) if s > 1 else None

        self.init_weights()

    def forward(self, x_in, children=None):
        # DLA_module의 s가 2 이상일 때, downsample과 project를 사용한다.
        if self.downsample is not None and self.project is not None:
            bottom = self.downsample(x_in)
            identity = self.project(bottom)
        else:
            identity = None

        x_1 = self.tree1(x_in, identity)
        x_2 = self.tree2(x_1)

        if children is None:
            x = torch.cat([x_1, x_2], dim=1)
        else:
            x = torch.cat([x_1, x_2, children], dim=1)

        x = self.root(x)
        
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)