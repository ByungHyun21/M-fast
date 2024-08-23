import os, sys, yaml, time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision

from src.layer import *
from src.backbone.model.dla.dla34 import dla34

class monouni_dla34(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.backbone = dla34(cfg)
    
    def postprocess(self, x):
        return x
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.postprocess(x)
        return x
    
    def load_pretrained_backbone(self, path):
        # self.backbone.load_state_dict(torch.load(path, map_location='cpu'))
        pass
    
    def load_pretrained_model(self, path):
        # self.load_state_dict(torch.load(path, map_location='cpu'))
        pass
        