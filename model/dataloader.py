import os
import cv2
import json
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset
import copy

class dataset(Dataset):
    def __init__(self, rank, cfg, purpose, preprocessor=None, augmentator=None):
        self.preprocessor = preprocessor
        self.augmentator = augmentator
        self.device = cfg['device']

        if purpose == 'Train':
            self.istrain = True
        if purpose == 'Valid':
            self.istrain = False
        
        self.annotations = dict()
        
        dataset_path = f"{cfg['dataset_root']}/{cfg['dataset']}"
        
        sub_dirs = os.listdir(f"{dataset_path}/{purpose}/Label")
        for sub_dir in sub_dirs:
            json_list = os.listdir(f"{dataset_path}/{purpose}/Label/{sub_dir}")
                
            if rank == 0:
                print(f"Load {dataset_path}/{purpose}/Label/{sub_dir}, {len(self.annotations)}")
                pbar = tqdm(json_list, ncols=0, leave=False)
            else:
                pbar = json_list
            
            
            for json_file in pbar:
                json_path = f"{dataset_path}/{purpose}/Label/{sub_dir}/{json_file}"
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                labels = dict()
                valid_labels = copy.deepcopy(json_data)
                valid_labels['object'] = []
                for obj in json_data['object']:
                    if obj['class'] not in cfg['network']['classes']:
                        continue

                    must_include = False
                    # check box2d exist
                    if 'box2d' in obj and 'box2d' in cfg['network']['task']:
                        must_include = True
                    
                    # check box3d exist
                    if 'box3d' in obj and 'box3d' in cfg['network']['task']:
                        must_include = True
                    
                    # check polygon exist
                    if 'polygon' in obj and 'polygon' in cfg['network']['task']:
                        must_include = True
                        
                    if must_include:
                        valid_labels['object'].append(obj)
                        
                if len(valid_labels['object']) == 0:
                    continue

                image_path = f"{dataset_path}/{purpose}/Image/{sub_dir}/{json_file.replace('.json', '.jpg')}"
                self.annotations[image_path] = valid_labels
            
                if cfg['test'] and len(self.annotations) > 100:
                    break
        
        self.annotation_keys, self.annotation_labels = list(self.annotations.keys()), list(self.annotations.values())

        
    def __len__(self):
        return len(self.annotation_keys)
    
    def __getitem__(self, idx):
        img, labels = self.read_data(idx)
    
        img, labels = self.augmentator(img, labels, self.istrain)
        
        if len(labels) == 0:
            print(self.annotation_keys[idx])
            return None, None
            
        gt = self.preprocessor(labels)

        return torch.from_numpy(img).permute(2, 0, 1).float().contiguous(), torch.from_numpy(gt).float().contiguous()
    
    def read_data(self, idx):
        img = cv2.imread(self.annotation_keys[idx])

        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        labels = self.annotation_labels[idx]
        
        return img, labels
