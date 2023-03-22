import os, platform
import yaml, time
import cv2
import random
from xml.etree.ElementTree import ElementTree as ET
from pathlib import Path

import numpy as np

import torch

from multiprocessing import Process, Pool, Queue
import psutil

class od_dataset():
    def __init__(self, config, ddp_rank, ddp_size, device, purpose, preprocessor=None, augmentator=None):
        self.ddp_rank = ddp_rank
        self.ddp_size = ddp_size
        self.dataset_root = Path(config['DATASET_PATH'])
        self.dataset = Path(config['DATASET'])
        self.names = config['CLASS']
        self.device = device

        if purpose == 'train':
            image_dir = self.dataset_root / self.dataset / Path('images_train')
            label_dir = self.dataset_root / self.dataset / Path('annotations_train')
            istrain = True
        if purpose == 'valid':
            image_dir = self.dataset_root / self.dataset / Path('images_valid')
            label_dir = self.dataset_root / self.dataset / Path('annotations_valid')
            istrain = False

        data = []
        for sub_dir in config['CATEGORY']:
            annos = os.listdir(label_dir / sub_dir)
            for anno in annos:
                label = str(label_dir / sub_dir / anno)
                image = str(image_dir / sub_dir / anno.replace('.xml', '.jpg'))
                data.append({'image':image, 'label':label})
        
        # check dataset
        valid = np.ones(len(data)).astype(np.bool8)
        for i in range(len(data)):
            isgood = self.check_data(data, i)
            
            if not isgood:
                valid[i] = False
        
        self.data = [item for keep, item in zip(valid, data) if keep]   
        
        random.shuffle(data)
        self.n_data = len(self.data)
        
        self.buffer = Queue(maxsize=config['BATCH'] * config['BUFFER_SIZE'])
        self.dataloader = dataloader(config, self.data, self.buffer, preprocessor, augmentator, ddp_rank, ddp_size, device, istrain)
        self.batchloader = batchloader(config, self.data, self.buffer)
        
    def check_data(self, data, index):
        error = False
        
        tree = ET().parse(data[index]['label'])

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
    
class dataloader():
    def __init__(self, config, data, buffer, preprocessor, augmentator, ddp_rank, ddp_size, device, istrain):
        self.data = data
        self.buffer = buffer
        self.preprocessor = preprocessor
        self.augmentator = augmentator
        self.ddp_rank = ddp_rank
        self.ddp_size = ddp_size
        self.device = device
        self.istrain = istrain
        
        self.names = config['CLASS']
        self.buffer_size = config['BATCH'] * config['BUFFER_SIZE']
        
        self.pool = Process(target=self.run, daemon=True)
        self.pool.start()
            
    def run(self):
        idxlist = list(range(len(self.data)))
        p = psutil.Process(os.getpid())
        p.nice(10)
        while True:
            random.shuffle(idxlist)
            
            for idx in idxlist:
                if self.buffer.qsize() > self.buffer_size*0.5:
                    time.sleep(0.01)
                    continue
                
                img, gt = self.get_item(idx)
                
                if img == None or gt == None:
                    continue
            
                self.buffer.put([img, gt])
            
            
    def get_item(self, index):
        img, labels, boxes = self.get_data(index)
    
        img, labels, boxes = self.augmentator(img, labels, boxes, self.istrain)
        
        if len(labels) == 0 or len(boxes) == 0:
            return None, None
            
        gt = self.preprocessor(labels, boxes)

        if self.ddp_size > 1:
            return torch.from_numpy(img).permute(2, 0, 1).to(self.ddp_rank).float(), torch.from_numpy(gt).to(self.ddp_rank).float()
        else:
            return torch.from_numpy(img).permute(2, 0, 1).to(self.device).float(), torch.from_numpy(gt).to(self.device).float()
        
    def get_data(self, index):
        
        tree = ET().parse(self.data[index]['label'])

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
        
        img = cv2.imread(self.data[index]['image'])

        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return img, labels, boxes
    
class batchloader():
    def __init__(self, config, data, buffer):
        self.data = data
        self.buffer = buffer
        self.batch = Queue(maxsize=config['BUFFER_SIZE'])
        self.batch_size = config['BATCH']
        
        self.pool = Process(target=self.run, daemon=True)
        self.pool.start()
        
    def run(self):
        p = psutil.Process(os.getpid())
        p.nice(10)
        while True:
            if self.buffer.qsize() < self.batch_size:    
                time.sleep(0.01)
                continue
            
            if self.batch.qsize() > self.batch_size * 0.5:
                time.sleep(0.01)
                continue
            
            batch_img = []
            batch_gt = []
            for i in range(self.batch_size):
                img, gt = self.buffer.get()
                
                batch_img.append(img)
                batch_gt.append(gt)
                
            batch_img = torch.stack(batch_img, dim=0)
            batch_gt = torch.stack(batch_gt, dim=0)
            
            self.batch.put([batch_img, batch_gt])