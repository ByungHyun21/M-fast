import os, platform
import yaml, time
import cv2
import random
from xml.etree.ElementTree import ElementTree as ET
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, config, purpose, preprocessor=None, augmentator=None):
        self.preprocessor = preprocessor
        self.augmentator = augmentator
        self.dataset = Path(config['DATASET'])
        self.names = config['CLASS']
        self.device = config['DEVICE']

        if purpose == 'train':
            image_dir = Path(config['DATASET_PATH']) / self.dataset / Path('images_train')
            label_dir = Path(config['DATASET_PATH']) / self.dataset / Path('annotations_train')
            self.istrain = True
        if purpose == 'valid':
            image_dir = Path(config['DATASET_PATH']) / self.dataset / Path('images_valid')
            label_dir = Path(config['DATASET_PATH']) / self.dataset / Path('annotations_valid')
            self.istrain = False

        data = []
        for sub_dir in config['CATEGORY']:
            annos = os.listdir(label_dir / sub_dir)
            for anno in annos:
                label = str(label_dir / sub_dir / anno)
                image = str(image_dir / sub_dir / anno.replace('.xml', '.jpg'))
                data.append({'image':image, 'label':label})
        
        #check dataset
        valid = np.ones(len(data)).astype(np.bool8)
        # for idx, sample in enumerate(data):
        #     isgood = self.check_data(sample)
            
        #     if idx % 10000 == 0:
        #         print(f'Data Validation Check : {idx} / {len(data)}, {idx*100/len(data):.2f} %')
            
        #     if not isgood:
        #         valid[idx] = False
        
        self.data = [item for keep, item in zip(valid, data) if keep]   
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, labels, boxes = self.read_data(idx)
    
        img, labels, boxes = self.augmentator(img, labels, boxes, self.istrain)
        
        if len(labels) == 0 or len(boxes) == 0:
            return None, None
            
        gt = self.preprocessor(labels, boxes)

        return torch.from_numpy(img).permute(2, 0, 1).float().contiguous(), torch.from_numpy(gt).float().contiguous()
    
    def read_data(self, idx):
        tree = ET().parse(self.data[idx]['label'])

        labels = []
        boxes = []
        path = tree.find('path').text
        filename = tree.find('filename').text
        objects = tree.findall('object')
        for obj in objects:
            label = self.names.index(obj.find('class').text)
            cx = float(obj.find('cx').text)
            cy = float(obj.find('cy').text)
            w = float(obj.find('w').text)
            h = float(obj.find('h').text)
            # labels.append([label])
            labels.append(label)
            boxes.append([cx, cy, w, h])
        
        img = cv2.imread(self.data[idx]['image'])

        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return img, labels, boxes
    
    def check_data(self, data):
        error = False
        
        tree = ET().parse(data['label'])

        objects = tree.findall('object')
        for obj in objects:
            try:
                label = self.names.index(obj.find('class').text)
                cx = float(obj.find('cx').text)
                cy = float(obj.find('cy').text)
                w = float(obj.find('w').text)
                h = float(obj.find('h').text)
            except:
                error = True
        
        if error:
            return False
        else:
            return True