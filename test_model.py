import argparse
import yaml
import time
import os
import cv2

import numpy as np
import torch

from tqdm import tqdm
from model.type.ssd.ssd_full import ssd_full
from model.type.ssd.anchor import anchor_generator
from model.utils import *

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# model.model = torch.load('runs/ssd_vgg16_voc0712/ssd_vgg16_voc0712_map_best.pth', map_location=config['DEVICE'])
def test(config:dict):
    # Select device
    if config['device'].lower() == 'cpu':
        config['DEVICE'] = 'cpu'
    else:
        config['DEVICE'] = 'cuda:0'   
    
    # weight load
    files = os.listdir(config['model_dir'])
    save_file = None
    for file in files:
        if config['best'] and file.endswith('best.pth'):
            save_file = file
        if config['last'] and file.endswith('last.pth'):
            save_file = file
            
    
    assert save_file is not None, 'best나 last를 선택해야 합니다.'
    
    config['DEVICE'] = 'cuda:0'
    os.environ['MASTER_ADDR'] = str(config['DDP_MASTER_ADDR'])
    os.environ['MASTER_PORT'] = str(config['DDP_MASTER_PORT'])
    
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    inner_model = torch.load(f"{config['model_dir']}/{save_file}", map_location=config['DEVICE'])
    if config['METHOD'] == 'ssd':
        anchor = anchor_generator(config)
        model = ssd_full(config, inner_model, anchor)
    
    model.model = model.model.to(config['DEVICE'])
    model.to(config['DEVICE'])
    
    model.model.eval()
    model.eval()
    
    w_input = config['INPUT_SIZE'][1]
    h_input = config['INPUT_SIZE'][0]
    w_output = 600
    h_output = 600
    
    # output
    vc_out = None
    if config['save_video'] is not None:
        vc_out = cv2.VideoWriter(f"{config['model_dir']}/{config['save_video']}", cv2.VideoWriter_fourcc(*'DIVX'), 30, (w_output, h_output))
    
    
    # cam
    if config['cam']:
        vc = cv2.VideoCapture(0)
        assert vc.isOpened(), '동영상 파일을 열 수 없습니다.'
        
        while True:
            ret, img = vc.read()
            if not ret:
                continue
            
            start = time.time()
            
            img = cv2.resize(img, (w_input, h_input))
            img_out = cv2.resize(img, (w_output, h_output))
            img = torch.from_numpy(img).permute([2, 0, 1]).unsqueeze(0).to(config['DEVICE'])
            pred = model(img)
            
            for i in range(len(config['TASK'])):
                if config['TASK'][i].lower() == 'object detection':
                    detections = pred[0]
                    # detections = [batch, class, score, x1, y1, x2, y2]
                    
                    for detection in detections:
                        if detection[1] > 0.5:
                            x1 = int(detection[2] * w_output)
                            y1 = int(detection[3] * h_output)
                            x2 = int(detection[4] * w_output)
                            y2 = int(detection[5] * h_output)
                            txt = f"{config['CLASS'][int(detection[0])]} : {detection[1]:.2f}"
                            
                            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            cv2.putText(img_out, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                            
            end = time.time()
            cv2.putText(img_out, f"FPS : {1/(end-start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            
            if config['show_result']:
                cv2.imshow('result', img_out)
                cv2.waitKey(1)
            
            if config['save_video'] is not None:
                vc_out.write(img_out)

    # img
    if config['img_dir'] is not None:
        img_list = os.listdir(config['img_dir'])
        img_list.sort()
        
        for img in img_list:
            start = time.time()
            
            img = cv2.imread(f"{config['img_dir']}/{img}")
            img = cv2.resize(img, (w_input, h_input))
            img_out = cv2.resize(img, (w_output, h_output))
            img = torch.from_numpy(img).permute([2, 0, 1]).unsqueeze(0).to(config['DEVICE'])
            pred = model(img)
            
            for i in range(len(config['TASK'])):
                if config['TASK'][i].lower() == 'object detection':
                    detections = pred[0]
                    # detections = [batch, class, score, x1, y1, x2, y2]
                    
                    for detection in detections:
                        if detection[1] > 0.5:
                            x1 = int(detection[2] * w_output)
                            y1 = int(detection[3] * h_output)
                            x2 = int(detection[4] * w_output)
                            y2 = int(detection[5] * h_output)
                            txt = f"{config['CLASS'][int(detection[0])]} : {detection[1]:.2f}"
                            
                            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            cv2.putText(img_out, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                            
            end = time.time()
            cv2.putText(img_out, f"FPS : {1/(end-start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            
            if config['show_result']:
                cv2.imshow('result', img_out)
                cv2.waitKey(1)
                
            if config['save_video'] is not None:
                vc_out.write(img_out)
        
        
    if config['save_video'] is not None:
        vc_out.release()
        
            
            
        
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='runs/ssd_mobilenet_v2_voc0712_decay')
    
    # Input
    parser.add_argument('--img_dir', default=None, type=str)
    parser.add_argument('--cam', action='store_true')
    
    # CPU or GPU
    parser.add_argument('--device', default='cuda:0', type=str)
    
    # Save Video (저장할 비디오 이름름)
    parser.add_argument('--save_video', default=None, type=str)
    
    # Select model weight
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--last', action='store_true')
    
    # Label Save (추론 결과를 저장할 것인지)
    parser.add_argument('--save_label', action='store_true')
    parser.add_argument('--label_dir', default=None, type=str) # label save dir
    
    # Show Result (추론 결과를 화면에 보여줄 것인지)
    parser.add_argument('--show_result', action='store_true') # GPU Server에서는 False 권장
    
    opt = parser.parse_args()
    
    opt.best = True
    opt.cam = True
    opt.show_result = True

    # read txt to dict
    with open(f"{opt.model_dir}/configuration.txt", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.update(vars(opt))

    test(config)