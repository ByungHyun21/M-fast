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
    
    def __init__(self, p):
        self.p = p
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        img = cv2.flip(img, 1)
        
        for obj in label['object']:
            if 'box2d' in obj:
                # cx flip
                obj['box2d']['cx'] = 1 - obj['box2d']['cx']
            if 'polygon' in obj:
                for poly in obj['polygon']:
                    for point in poly:
                        point[0] = 1 - point[0]
            
        return img, label
