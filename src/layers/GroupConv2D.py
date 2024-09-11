from typing import Union, Optional
from .utils import *

import torch
import torch.nn as nn
import torch.nn.init as init

class GroupConv2D(nn.Module):
    """
    Group Convolution Layer
    
    cin : input channel
    cout : output channel
    k : kernel size
    s : stride
    d : dilation
    bias : bias
    pm : padding mode
    act : activation function
    """
    def __init__(self, 
                 cin:int, 
                 cout:int, 
                 k:int=3, 
                 s:int=1, 
                 p:Union[int, None]=None, 
                 d:int=1, 
                 bn:Union[list, None]=['bn'], 
                 bias:bool=True, 
                 pm:str='zeros', 
                 act:Union[str, None]='relu'):
        super().__init__()
        assert(cout % cin == 0) # cout must be multiple of cin
        
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        
        self.conv2d = nn.Conv2d(cin, cout, k, s, p, d, cin, bias, pm)
        
        self.bn = None
        if bn == 'bn':
            self.bn = nn.BatchNorm2d(cout)
        
        self.act = None
        if act == 'relu6':
            self.act = nn.ReLU6()
        if act == 'relu':
            self.act = nn.ReLU()
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU()

        self.init_weights()

    def forward(self, x):
        x = self.conv2d(x)
        
        if self.bn is not None:
            x = self.bn(x.contiguous())

        if self.act is not None:
            x = self.act(x)

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)