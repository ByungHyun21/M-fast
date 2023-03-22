import numpy as np
from torchvision.models import mobilenet_v2, mobilenet_v3_small

import torchvision.transforms as transforms
from PIL import Image
import torch

net = mobilenet_v2(weights='IMAGENET1K_V2')

