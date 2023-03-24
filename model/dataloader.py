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

class dataset():
    def __init__(self, config, ddp_rank, ddp_size, device, purpose, preprocessor=None, augmentator=None):
        self.ddp_rank = ddp_rank
        self.ddp_size = ddp_size
        self.dataset = Path(config['DATASET'])
        self.names = config['CLASS']
        self.device = device

        if purpose == 'train':
            image_dir = '/dataset' / self.dataset / Path('images_train')
            label_dir = '/dataset' / self.dataset / Path('annotations_train')
            istrain = True
        if purpose == 'valid':
            image_dir = '/dataset' / self.dataset / Path('images_valid')
            label_dir = '/dataset' / self.dataset / Path('annotations_valid')
            istrain = False

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
        
        random.shuffle(data)
        self.n_data = len(self.data)
        self.steps_per_epoch = self.n_data // config['BATCH_SIZE']
        
        self.buffer = Queue(maxsize=config['BATCH_PER_GPU'] * config['BUFFER_SIZE'])
        self.batch = Queue(maxsize=config['BUFFER_SIZE'])
        self.dataloader = dataloader(config, self.data, self.buffer, preprocessor, augmentator, ddp_rank, ddp_size, device, istrain)
        self.batchloader = batchloader(config, self.buffer, self.batch)
    
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
    
class dataloader():
    def __init__(self, config, data, buffer, preprocessor, augmentator, ddp_rank, ddp_size, device, istrain):
        self.preprocessor = preprocessor
        self.augmentator = augmentator
        self.ddp_rank = ddp_rank
        self.ddp_size = ddp_size
        self.device = device
        self.istrain = istrain
        
        self.names = config['CLASS']
        self.buffer_size = config['BATCH_PER_GPU'] * config['BUFFER_SIZE']
        
        self.pool = []
        for _ in range(config['WORKERS']):
            self.pool.append(Process(target=self.run, args=(data, buffer), daemon=True))
            self.pool[-1].start()
            
    def run(self, data, buffer):
        idxlist = list(range(len(data)))
        
        while True:
            random.shuffle(idxlist)
            
            for idx in idxlist:
                img, gt = self.get_item(data[idx])
                
                if img == None or gt == None:
                    continue
            
                buffer.put([img, gt])
            
            
    def get_item(self, data):
        img, labels, boxes = self.get_data(data)
    
        img, labels, boxes = self.augmentator(img, labels, boxes, self.istrain)
        
        if len(labels) == 0 or len(boxes) == 0:
            return None, None
            
        gt = self.preprocessor(labels, boxes)

        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(gt).float()
        
    def get_data(self, data):
        tree = ET().parse(data['label'])

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
        
        img = cv2.imread(data['image'])

        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return img, labels, boxes
    
class batchloader():
    def __init__(self, config, buffer, batch):
        self.batch_size = config['BATCH_PER_GPU']
        self.pool = Process(target=self.run, args=(buffer, batch), daemon=True)
        self.pool.start()
        
    def run(self, buffer, batch):
        while True:
            batch_img = []
            batch_gt = []
            for i in range(self.batch_size):
                img, gt = buffer.get()
                
                batch_img.append(img)
                batch_gt.append(gt)
                
            batch_img = torch.stack(batch_img, dim=0)
            batch_gt = torch.stack(batch_gt, dim=0)
            
            batch.put([batch_img, batch_gt])