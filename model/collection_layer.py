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

class conv2d_bn_act(nn.Module):
    def __init__(self, inc, outc, kernel=(3, 3), act=None, strides=(1, 1), padding=1, bias=False):
        super().__init__()
        if kernel == (1, 1):
            padding = 0

        self.conv2d = nn.Conv2d(inc, outc, kernel, strides, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outc)
        
        self.act = None
        if act == 'relu6':
            self.act = nn.ReLU6()
        if act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x

class depth2d_bn_act(nn.Module):
    def __init__(self, inc, outc, act=None, strides=(1, 1), bias=False):
        super().__init__()

        self.depth2d = nn.Conv2d(inc, outc, (3, 3), strides, padding=1, groups=inc, bias=bias)
        self.bn = nn.BatchNorm2d(outc)
        
        self.act = None
        if act == 'relu6':
            self.act = nn.ReLU6()
        if act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.depth2d(x)
        x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x

class Mobilenet_v2_conv(nn.Module):
    def __init__(self, inc, outc, strides=(1, 1), bias=False):
        super().__init__()
        self.residual = False
        if inc == outc:
            self.residual = True

        self.conv2d_1 = conv2d_bn_act(inc, inc*6, (1, 1), 'relu6', bias=bias)
        self.depthconv = depth2d_bn_act(inc*6, inc*6, 'relu6', strides=strides, bias=bias)
        self.conv2d_2 = conv2d_bn_act(inc*6, outc, (1, 1), None, bias=bias)

    def forward(self, x_in):
        x = self.conv2d_1(x_in)
        x = self.depthconv(x)
        x = self.conv2d_2(x)

        if self.residual:
            x = x + x_in

        return x

