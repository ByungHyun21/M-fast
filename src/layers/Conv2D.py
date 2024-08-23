from typing import Union, Optional
from .utils import *

import torch.nn as nn

class Conv2D(nn.Module):
    """
    일반적인 Convolution Layer
    
    cin : input channel
    cout : output channel
    k : kernel size
    s : stride
    p : padding
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
                 bn:Union[str, None]='bn', 
                 bias:bool=True, 
                 pm:str='zeros', 
                 act:Union[str, None]='relu'):
        super().__init__()
        
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        
        self.conv2d = nn.Conv2d(cin, cout, k, s, p, d, 1, bias, pm)
        
        self.bn = None
        if bn is None:
            pass
        elif bn.lower() == 'bn':
            self.bn = nn.BatchNorm2d(cout)
        elif bn.lower() == 'gn':
            self.bn = nn.GroupNorm(bn[1], cout)
        
        self.act = None
        if act is None:
            pass
        elif act.lower() == 'relu6':
            self.act = nn.ReLU6()
        elif act.lower() == 'relu':
            self.act = nn.ReLU()

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