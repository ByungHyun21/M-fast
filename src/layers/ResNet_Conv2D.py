from typing import Union, Optional
from .utils import *
    
import torch.nn as nn

class ResNet_Conv2d(nn.Module):
    """
    Residual Convolution Layer
    
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
        
        if cin == cout and s == 1:
            self.residual = True
        else:
            self.residual = False

        self.conv2d_1 = nn.Conv2d(cin, cout, k, s, p, d, 1, bias, pm)
        self.conv2d_2 = nn.Conv2d(cout, cout, k, 1, p, d, 1, bias, pm)

        self.bn1 = None
        self.bn2 = None
        if bn is None:
            pass
        elif bn.lower() == 'bn':
            self.bn1 = nn.BatchNorm2d(cout)
            self.bn2 = nn.BatchNorm2d(cout)
        elif bn.lower() == 'gn':
            self.bn1 = nn.GroupNorm(bn[1], cout)
            self.bn2 = nn.GroupNorm(bn[1], cout)
        
        self.act1 = None
        self.act2 = None
        if act is None:
            pass
        elif act.lower() == 'relu6':
            self.act1 = nn.ReLU6()
            self.act2 = nn.ReLU6()
        elif act.lower() == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()

        self.init_weights()

    def forward(self, x_in, identity=None):
        x = self.conv2d_1(x_in)        
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.act1(x) if self.act1 is not None else x

        x = self.conv2d_2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        if identity is not None:
            x = x + identity
        else:
            x = x + x_in if self.residual else x
        x = self.act2(x) if self.act2 is not None else x
        
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)