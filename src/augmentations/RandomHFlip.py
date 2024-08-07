import random
import cv2
import numpy as np
import math
import copy

class RandomHFlip(object):
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
        
        img = cv2.flip(img, 0)
        
        for obj in label['object']:
            if 'box2d' in obj:
                # cy flip
                obj['box2d']['cy'] = 1 - obj['box2d']['cy']
            if 'polygon' in obj:
                for poly in obj['polygon']:
                    for point in poly:
                        point[1] = 1 - point[1]
            
        return img, label
