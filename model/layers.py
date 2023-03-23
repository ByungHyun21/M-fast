import torch
import torch.nn as nn
import torch.nn.init as init

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(module):
    # # # # #
    # Conv2D
    if isinstance(module, nn.Conv2d):
        xavier(module.weight.data) 
        if module.bias is not None:
            module.bias.data.zero_()

    # # # # #
    # Linear
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()

def autopad(k, p=None):
    # Calculate padding for 'same' convolution
    # k = 1, p = 0
    # k = 2, p = 1
    # k = 3, p = 1
    # k = 4, p = 2
    # k = 5, p = 2
    # k = 6, p = 3 ...
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return k // 2

class Conv2d(nn.Module):
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
    def __init__(self, cin:int, cout:int, k:int=3, s:int=1, d:int=1, bn:bool=True, bias:bool=True, pm:str='zeros', act:str='relu'):
        super().__init__()
        # default 
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv2d = nn.Conv2d(cin, cout, k, s, autopad(k, p=None), d, 1, bias, pm)
        
        # default
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(cout)
        
        self.act = None
        if act == 'relu6':
            self.act = nn.ReLU6()
        if act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        
        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x

class GroupConv2d(nn.Module):
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
    def __init__(self, cin:int, cout:int, k:int=3, s:int=1, d:int=1, bn:bool=True, bias:bool=True, pm:str='zeros', act:str='relu'):
        super().__init__()
        assert(cout % cin == 0) # cout must be multiple of cin
        
        # default 
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.depth2d = nn.Conv2d(cin, cout, k, s, autopad(k, p=None), d, cin, bias, pm)
        
        # default
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(cout)
        
        self.act = None
        if act == 'relu6':
            self.act = nn.ReLU6()
        if act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.depth2d(x)
        
        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x

class MobileNet_V2_Conv2d(nn.Module):
    """
    cin : input channel
    cout : output channel
    s : stride
    bias : bias
    """
    def __init__(self, cin:int, cout:int, s:int=1, bn:bool=True, bias:bool=False):
        super().__init__()
        self.residual = False
        if cin == cout and s == 1:
            # cin == cout : residual connection (same channel)
            # s == 1 : residual connection (no stride)
            self.residual = True

        self.Conv2d_1 = Conv2d(cin, cin*6, 1, bn=bn, bias=bias, act='relu6')
        self.GroupConv2d = GroupConv2d(cin*6, cin*6, s=s, bn=bn, bias=bias, act='relu6')
        self.Conv2d_2 = Conv2d(cin*6, cout, 1, bn=bn, bias=bias, act=None)

    def forward(self, x_in):
        x = self.Conv2d_1(x_in)
        x = self.GroupConv2d(x)
        x = self.Conv2d_2(x)

        if self.residual:
            x = x + x_in

        return x

