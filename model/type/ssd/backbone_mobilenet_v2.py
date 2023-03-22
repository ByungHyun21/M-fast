import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from model.collection_layer import *

class Mobilenet_v2_from_torchvision(nn.Module):
    def __init__(self):
        super().__init__()
        
        backbone = mobilenet_v2(weights='IMAGENET1K_V2')

        self.backbone = backbone.features[:14] # feature 0 ~ feature 13

    def forward(self, x):
        return self.backbone(x)

class Mobilenet_v2_from_scratch(nn.Module):
    def __init__(self, config):
        super().__init__()
        bias = config['BIAS']
        
        self.feature = nn.Sequential(
            conv2d_bn_act(3, 32, (3, 3), 'relu6', strides=(2, 2), bias=bias), # 300x300 -> 150x150
            depth2d_bn_act(32, 32, 'relu6', bias=bias),
            conv2d_bn_act(32, 16, (1, 1), bias=bias),
            
            Mobilenet_v2_conv(16, 24, strides=(2, 2), bias=bias), # 150x150 -> 75x75
            Mobilenet_v2_conv(24, 24, bias=bias),
            
            Mobilenet_v2_conv(24, 32, strides=(2, 2), bias=bias), # 75x75 -> 38x38
            Mobilenet_v2_conv(32, 32, bias=bias),
            Mobilenet_v2_conv(32, 32, bias=bias),
            
            Mobilenet_v2_conv(32, 64, strides=(2, 2), bias=bias), # 38x38 -> 19x19
            Mobilenet_v2_conv(64, 64, bias=bias),
            Mobilenet_v2_conv(64, 64, bias=bias),
            Mobilenet_v2_conv(64, 64, bias=bias),
            Mobilenet_v2_conv(64, 96, bias=bias),
            Mobilenet_v2_conv(96, 96, bias=bias),
            Mobilenet_v2_conv(96, 96, bias=bias),
        )
        
    def forward(self, x):
        return self.feature(x)


class Mobilenet_v2_extra(nn.Module):
    def __init__(self, config):
        super().__init__()
        bias = config['BIAS']
        
        self.layer1 = nn.Sequential(
            conv2d_bn_act(96, 576, (1, 1), 'relu6', bias=bias)
        )

        self.layer2 = nn.Sequential(
            depth2d_bn_act(576, 576, 'relu6', strides=(2,2), bias=bias),
            conv2d_bn_act(576, 160, (1, 1), None, bias=bias),
            Mobilenet_v2_conv(160, 160, bias=bias),
            Mobilenet_v2_conv(160, 160, bias=bias),
            Mobilenet_v2_conv(160, 320, bias=bias),
            conv2d_bn_act(320, 1280, (1, 1), 'relu6', bias=bias)
        )

        self.layer3 = nn.Sequential(
            conv2d_bn_act(1280, 256, (1, 1), 'relu6', bias=bias),
            depth2d_bn_act(256, 256, 'relu6', strides=(2, 2), bias=bias),
            conv2d_bn_act(256, 512, (1, 1), 'relu6', bias=bias)
        )

        self.layer4 = nn.Sequential(
            conv2d_bn_act(512, 128, (1, 1), 'relu6', bias=bias),
            depth2d_bn_act(128, 128, 'relu6', strides=(2, 2), bias=bias),
            conv2d_bn_act(128, 256, (1, 1), 'relu6', bias=bias)
        )

        self.layer5 = nn.Sequential(
            conv2d_bn_act(256, 128, (1, 1), 'relu6', bias=bias),
            depth2d_bn_act(128, 128, 'relu6', strides=(2, 2), bias=bias),
            conv2d_bn_act(128, 256, (1, 1), 'relu6', bias=bias)
        )

        self.layer6 = nn.Sequential(
            conv2d_bn_act(256, 64, (1, 1), 'relu6', bias=bias),
            depth2d_bn_act(64, 64, 'relu6', strides=(2, 2), bias=bias),
            conv2d_bn_act(64, 128, (1, 1), 'relu6', bias=bias)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        return (x1, x2, x3, x4, x5, x6)

