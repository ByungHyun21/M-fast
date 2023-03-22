import argparse
import os
import yaml
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from model.collection_layer import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/mv2_test.yaml', help='plan has training info like epoch, batch_size, ... etc')
opt = parser.parse_args()
with open(opt.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
# # # # #
# torch, CUDA, gpu Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

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
    
def train(ddp_rank):
    # model = Mobilenet_v2_from_torchvision()
    model = Mobilenet_v2_from_scratch(config)
    model.to('cuda')
    model.eval()
    torch.onnx.export(model, torch.rand(1, 3, 300, 300), 'debug/model/model.onnx', opset_version=10, export_params=True)
    print('done')


if __name__ == '__main__':
    train(0)
    # mp.spawn(train, nprocs=torch.cuda.device_count(), join=True)