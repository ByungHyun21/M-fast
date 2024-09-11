import random
import cv2
import numpy as np
import math
import copy

class FlipWithOpticalCenter(object):
    def __init__(self, p=1.0):
        self.p = p
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img:np.array, label):
        if random.random() > self.p:
            return img, label
        
        label['flag'].append('FlipWithOpticalCenter')
        
        
        
        
        if 'ObjectDepthBoxGeneration' in label['flag']:
            pass
            
        return img, label
    