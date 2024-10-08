import random
import cv2
import numpy as np
import math
import copy

class ImageNormalization(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0):
        self.p = p
        self.mean = mean
        self.std = std
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img:np.array, label):
        if random.random() > self.p:
            return img, label
        
        # TODO:
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
            
        return img, label
