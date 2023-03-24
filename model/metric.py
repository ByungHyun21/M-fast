import os
import cv2
from pathlib import Path
from xml.etree.ElementTree import ElementTree as ET

import torch

import numpy as np
from tqdm import tqdm

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
        
        self.valid_class = []
        for c in self.target_class:
            if c in model.target_class:
                self.valid_class.append(c)
                
        self.table_model = []
        for c in model.target_class: # Model to valid_class Mapping
            if c in self.valid_class:
                self.table_model.append(self.valid_class.index(c))
            else:
                self.table_model.append(-1)
                
        self.table_dataset = []
        for c in self.target_class: # Dataset to valid_class Mapping
            if c in self.valid_class:
                self.table_dataset.append(self.valid_class.index(c))
            else:
                self.table_dataset.append(-1)
        
        data = []
        for sub_dir in self.category:
            annos = os.listdir(label_dir / sub_dir)
            for anno in annos:
                label = str(label_dir / sub_dir / anno)
                image = str(image_dir / sub_dir / anno.replace('.xml', '.jpg'))
                data.append({'image':image, 'label':label})
        
        #check dataset
        log('check dataset')
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
        calculators = []
        calculators.append(mAP_calculator(0, 1, 0.5))                   # mAP 0.5
        calculators.append(mAP_calculator(0, 1, 0.55))                  # mAP 0.55
        calculators.append(mAP_calculator(0, 1, 0.6))                   # mAP 0.6
        calculators.append(mAP_calculator(0, 1, 0.65))                  # mAP 0.65
        calculators.append(mAP_calculator(0, 1, 0.7))                   # mAP 0.7
        calculators.append(mAP_calculator(0, 1, 0.75))                  # mAP 0.75
        calculators.append(mAP_calculator(0, 1, 0.8))                   # mAP 0.8
        calculators.append(mAP_calculator(0, 1, 0.85))                  # mAP 0.85
        calculators.append(mAP_calculator(0, 1, 0.9))                   # mAP 0.9
        calculators.append(mAP_calculator(0, 1, 0.95))                  # mAP 0.95
        calculators.append(mAP_calculator(0, (1/6)**2, 0.5))            # mAP small
        calculators.append(mAP_calculator((1/6)**2, (1/3)**2, 0.6))     # mAP medium
        calculators.append(mAP_calculator((1/3)**2, 1, 0.7))            # mAP large
        
        log(self.dataset + ' : mAP Calculation')
        pbar = tqdm(self.data, desc=self.dataset + ' : mAP Calculation', ncols=0)
        for d in pbar:
            img, label, box = self.read_data(d)
            
            pred = model(img.to(model.device))
            
        
        
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
                
                if label in self.valid_class:
                    valid = True
            except:
                error = True
                
        if error:
            return False
        
        if valid:
            return True
        else:
            return False
        
    def read_data(self, data):
        label = []
        box = []
        
        tree = ET().parse(data['label'])

        objects = tree.findall('object')
        for obj in objects:
            l = obj.find('class').text
            cx = float(obj.find('cx').text)
            cy = float(obj.find('cy').text)
            w = float(obj.find('w').text)
            h = float(obj.find('h').text)
            
            if l not in self.target_class:
                continue
            
            #cx, cy, w, h -> x1, y1, x2, y2
            x1 = cx - w/2
            x2 = cx + w/2
            y1 = cy - h/2
            y2 = cy + h/2
            
            label.append(self.target_class.index(l))
            box.append([x1, y1, x2, y2])
            
        img = cv2.imread(data['image'])
            
        return img, label, box
            
        
class mAP_calculator(object):
    def __init__(self, min_area, max_area, iou_threshold):
        super().__init__()
        
        
    def __call__(self):
        
        pass