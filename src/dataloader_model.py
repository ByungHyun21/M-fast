import os
import cv2
import json
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset
import copy

class dataloader_model(Dataset):
    def __init__(self, rank, cfg, purpose, augmentator=None):
        self.augmentator = augmentator
        self.device = cfg['device']
        self.valid_classes = cfg['classes']
        
        self.istrain = None
        if purpose == 'Train':
            self.istrain = True
        if purpose == 'Valid':
            self.istrain = False
        assert self.istrain is not None, f"purpose should be 'Train' or 'Valid', but got {purpose}"
        
        self.annotations = dict()
        
        dataset_path = f"{cfg['dataset_root']}/{cfg['dataset']}"
        
        sub_dirs = os.listdir(f"{dataset_path}/{purpose}/Camera/Label")
        for sub_dir in sub_dirs:
            json_list = os.listdir(f"{dataset_path}/{purpose}/Camera/Label/{sub_dir}")
                
            if rank == 0:
                print(f"Load {dataset_path}/{purpose}/Camera/Label/{sub_dir}, {len(self.annotations)}")
                pbar = tqdm(json_list, ncols=0, leave=False)
            else:
                pbar = json_list
            
            
            for json_file in pbar:
                this_data_valid = False
                
                image_path = f"{dataset_path}/{purpose}/Camera/Image/{sub_dir}/{json_file.replace('.json', '.jpg')}"
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                json_path = f"{dataset_path}/{purpose}/Camera/Label/{sub_dir}/{json_file}"
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                labels = dict()
                valid_labels = copy.deepcopy(json_data)
                valid_labels['objects'] = []
                for obj in json_data['objects']:
                    if obj['class'] not in self.valid_classes:
                        continue

                    for t in cfg['task']:
                        if t not in obj:
                            continue

                    if 'box3d' in obj:
                        if 'intrinsic' not in json_data:
                            continue
                        
                        fx = float(json_data['intrinsic']['fx'])
                        fy = float(json_data['intrinsic']['fy'])
                        cx = float(json_data['intrinsic']['cx'])
                        cy = float(json_data['intrinsic']['cy'])
                        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        
                        tx = float(obj['box3d']['translation']['x'])
                        ty = float(obj['box3d']['translation']['y'])
                        tz = float(obj['box3d']['translation']['z'])
                        center = np.array([tx, ty, tz]).reshape(3, 1)
                        
                        if tz < 0:
                            continue
                        
                        # project 3d center to 2d
                        center = intrinsic @ center
                        center = center / center[2]
                        
                        imh, imw = image.shape[:2]
                        
                        # Check if the center is in the image
                        if center[0] < 0 or center[0] >= imw or center[1] < 0 or center[1] >= imh:
                            continue
                        
                    this_data_valid = True
                        
                if not this_data_valid:
                    continue

                image_path = f"{dataset_path}/{purpose}/Camera/Image/{sub_dir}/{json_file.replace('.json', '.jpg')}"
                self.annotations[image_path] = json_path
            
                if cfg['test'] and len(self.annotations) > 500:
                    break
        
        self.annotation_keys, self.annotation_labels = list(self.annotations.keys()), list(self.annotations.values())
        
        self.cfg = cfg

        
    def __len__(self):
        return len(self.annotation_keys)
    
    def __getitem__(self, idx):
        img, labels = self.read_data(idx)
    
        img, labels = self.augmentator(img, labels, self.istrain)
        
        if len(labels) == 0:
            print(self.annotation_keys[idx])
            return None, None
        
        return torch.from_numpy(img).permute(2, 0, 1).float().contiguous(), labels
    
    def read_data(self, idx):
        img = cv2.imread(self.annotation_keys[idx])
        imh, imw = img.shape[:2]
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        with open(self.annotation_labels[idx], 'r') as f:
            labels = json.load(f)
            
        labels['image_path'] = self.annotation_keys[idx]
        labels['label_path'] = self.annotation_labels[idx]
        
        if 'intrinsic' in labels:
            fx = labels['intrinsic']['fx']
            fy = labels['intrinsic']['fy']
            cx = labels['intrinsic']['cx']
            cy = labels['intrinsic']['cy']
            
            labels['intrinsic'] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
        if 'extrinsic' in labels:
            rotation = np.array(labels['extrinsic']['rotation']).reshape(3, 3)
            tx = labels['extrinsic']['translation']['x']
            ty = labels['extrinsic']['translation']['y']
            tz = labels['extrinsic']['translation']['z']
            translation = np.array([tx, ty, tz]).reshape(3, 1)
            
            labels['extrinsic'] = np.vstack([np.hstack([rotation, translation]), np.array([0, 0, 0, 1])])
                        
        valid_objects = []
        for obj in labels['objects']:
            if obj['class'] not in self.valid_classes:
                continue
            
            for t in self.cfg['task']:
                if t not in obj:
                    continue
            
            if 'box3d' in obj:
                rotation = np.array(obj['box3d']['rotation']).reshape(3, 3)
                tx = obj['box3d']['translation']['x']
                ty = obj['box3d']['translation']['y']
                tz = obj['box3d']['translation']['z']
                translation = np.array([tx, ty, tz]).reshape(3, 1)
                
                if tz < 0:
                    continue
                
                # project 3d center to 2d
                center = np.array([tx, ty, tz]).reshape(3, 1)
                center = labels['intrinsic'] @ center
                center = center / center[2]
            
                # Check if the center is in the image
                if center[0] < 0 or center[0] >= imw or center[1] < 0 or center[1] >= imh:
                    continue
                
                obj['box3d']['extrinsic'] = np.vstack([np.hstack([rotation, translation]), np.array([0, 0, 0, 1])])
            
            valid_objects.append(obj)
            
        labels['objects'] = valid_objects
        
        return img, labels
