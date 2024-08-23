from typing import Union, Optional
from .utils import *

import torch
import torch.nn as nn
import torch.nn.init as init

class MaxPool2D(nn.Module):
    """
    Max Pooling Layer
    
    k : kernel size
    s : stride
    p : padding
    """
    def __init__(self, 
                 k:int=3, 
                 s:int=1, 
                 p:Union[int, None]=1):
        super().__init__()
        
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
            
        self.maxpool2d = nn.MaxPool2d(k, s, p, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool2d(x)
        return x