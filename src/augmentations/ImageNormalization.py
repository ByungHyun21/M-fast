import random
import cv2
import numpy as np
import math
import copy

class RandomVFlip(object):
    # box2d : change
    # box3d : change (TODO)
    # polygon : change
    # keypoint : change (TODO)
    
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0):
        self.p = p
        self.mean = mean
        self.std = std
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        # TODO:
            
        return img, label
