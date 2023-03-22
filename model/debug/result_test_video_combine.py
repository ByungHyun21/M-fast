import argparse
import os, yaml, cv2, time
import numpy as np
from tqdm import tqdm

import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.network_manager import network_manager
from model.od_ssd import ssd_post

models = ['ssd_argoseye_0302', 'ssd_argoseye_woIN_0302', 'ssd_coco_80_0302', 'ssd_coco_80_woIN_0302']
# category = ['CH1', 'CH2', 'CH3', 'CH4']
category = ['IR_1', 'IR_2', 'IR_3', 'IR_4']

for video_category in category:

    video_file_name = os.path.join('runs', video_category + '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    vw = 1200
    vh = 1200

    video = cv2.VideoWriter(video_file_name, fourcc, 30, (vw, vh))

    vc = []
    for model in models:
        vc.append(cv2.VideoCapture(os.path.join('runs', model, video_category + '.mp4')))
        
    print(video_file_name)
        
    isend = False
    while True:
        imgs = []
        for i, vc_ in enumerate(vc):
            ret, img = vc_.read()
            
            cv2.putText(img, models[i], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            imgs.append(img)
            if ret == False:
                isend = True
                break
        
        if isend:
            break
        
        if len(imgs) == 4:
            img1 = np.concatenate((imgs[0], imgs[1]), axis=1)
            img2 = np.concatenate((imgs[2], imgs[3]), axis=1)
            img = np.concatenate((img1, img2), axis=0)
            
        img = cv2.resize(img, (vw, vh))
        video.write(img)
    video.release()