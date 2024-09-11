import random
import cv2
import numpy as np
import math
import copy

class ObjectDepthBoxGeneration(object):
    def __init__(self, p=1.0):
        self.p = p
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img:np.array, label):
        if random.random() > self.p:
            return img, label
        
        label['flag'].append('ObjectDepthBoxGeneration')
        
        
            
        return img, label
    
