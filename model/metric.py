import os
import cv2
from pathlib import Path
from xml.etree.ElementTree import ElementTree as ET

import torch

import numpy as np

from model.utils import log

class mAP(object):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.category = None
        self.target_class = None
        
    def set(self, dataset, category, target_class):
        self.dataset = dataset
        self.category = category
        self.target_class = target_class
        
    def __call__(self, model):
        assert self.dataset is not None, "dataset is not set"
        assert self.category is not None, "category is not set"
        assert self.target_class is not None, "target_class is not set"
        
        # mAP Calculation
        # model output = [batch, [class, score, x1, y1, x2, y2]]
        
        image_dir = Path('/dataset') / self.dataset / Path('images_valid')
        label_dir = Path('/dataset') / self.dataset / Path('annotations_valid')
        
        valid_class = []
        for c in self.target_class:
            if c in model.target_class:
                valid_class.append(c)
        
        data = []
        for sub_dir in self.category:
            annos = os.listdir(label_dir / sub_dir)
            for anno in annos:
                label = str(label_dir / sub_dir / anno)
                image = str(image_dir / sub_dir / anno.replace('.xml', '.jpg'))
                data.append({'image':image, 'label':label})
        
        #check dataset
        valid = np.ones(len(data)).astype(np.bool8)
        for idx, sample in enumerate(data):
            isgood = self.check_data(sample)    
            if not isgood:
                valid[idx] = False
        
        self.data = [item for keep, item in zip(valid, data) if keep]   
        
        #mAP Calculation
        # mAP 0.5:0.05:0.95
        # mAP 0.5
        # mAP 0.75
        # mAP small (object area < (1/6)^2))
        # mAP medium (object area > (1/6)^2) and (object area < (1/3)^2)
        # mAP large (object area > (1/3)^2))
        
        
        
        pass
    
    def check_data(self, data):
        # check label
        # model.target_class == mAP.target_class
        error = False
        valid = False
        
        tree = ET().parse(data['label'])

        objects = tree.findall('object')
        for obj in objects:
            try:
                label = obj.find('class').text
                cx = float(obj.find('cx').text)
                cy = float(obj.find('cy').text)
                w = float(obj.find('w').text)
                h = float(obj.find('h').text)
                
                if label in self.target_class:
                    valid = True
            except:
                error = True
                
        if error:
            return False
        
        if valid:
            return True
        else:
            return False
        
        
        
