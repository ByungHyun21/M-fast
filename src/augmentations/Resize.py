import random
import cv2
import numpy as np
import math
import copy

class Resize(object):
    # box2d : no change
    # box3d : no change
    # polygon : no change
    # keypoint : no change
    def __init__(self, size):
        self.imh = size[0]
        self.imw = size[1]
        
    def __call__(self, img, label):
        img = cv2.resize(img, (self.imw, self.imh))
        
        return img, label