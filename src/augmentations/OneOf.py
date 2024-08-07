import random
import cv2
import numpy as np
import math
import copy

class OneOf(object):
    def __init__(self, auglist, p=0.5):
        self.p = p
        self.auglist = auglist
        self.na = len(auglist) - 1
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        idx = random.randint(0, self.na)
        
        img, label = self.auglist[idx](img, label)
            
        return img, label
