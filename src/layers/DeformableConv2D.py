from typing import Union, Optional
from .utils import *

import torch
import torch.nn as nn
import torchvision.ops

class DeformableConv2D(nn.Module):
    """
    Deformable Convolution Layer (v2)
    
    cin : input channel
    cout : output channel
    k : kernel size
    s : stride
    p : padding
    d : dilation
    bn : batch normalization
    bias : bias
    pm : padding mode
    act : activation function 
    """
    def __init__(self, 
                 cin:int, 
                 cout:int, 
                 k:int=4, 
                 s:int=2, 
                 p:Union[int, None]=None, 
                 d:int=1, 
                 bn:Union[list, None]=['bn'], 
                 bias:bool=True, 
                 pm:str='zeros', 
                 act:Union[str, None]='relu'):
        super().__init__()

        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]

        self.offset_conv = nn.Conv2d(cin, k * k * 2, k, s, p, d, 1, bias, pm)
        self.modulator_conv = nn.Conv2d(cin, k * k, k, s, p, d, 1, bias, pm)
        self.regular_conv = nn.Conv2d(cin, cout, k, s, p, d, 1, bias, pm)
        
        self.bn = None
        if bn is None:
            pass
        elif bn[0].lower() == 'bn':
            self.bn = nn.BatchNorm2d(cout)
        elif bn[0].lower() == 'gn':
            self.bn = nn.GroupNorm(bn[1], cout)
        
        self.act = None
        if act is None:
            pass
        elif act.lower() == 'relu6':
            self.act = nn.ReLU6()
        elif act.lower() == 'relu':
            self.act = nn.ReLU()
        elif act.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU()

        self.init_weights()

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2 * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(x, 
                              offset=offset, 
                              mask=modulator, 
                              weight=self.regular_conv.weight, 
                              bias=self.regular_conv.bias, 
                              stride=self.regular_conv.stride, 
                              padding=self.regular_conv.padding, 
                              dilation=self.regular_conv.dilation)

        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)