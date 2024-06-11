import argparse
import json
import time
import os
import cv2

import numpy as np
import random

from tqdm import tqdm
from model.type.ssd.ssd_full import ssd_full
from model.type.ssd.anchor import anchor_generator
from model.network import network

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# TODO:
# 1. CPU에서 동작하도록 수정

random.seed(100)
colormap = []
for i in range(1000):
    colormap.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

@torch.no_grad()
def test(config:dict):
    # Select device
    if config['run_cpu']:
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
    
    # # config['DEVICE'] = 'cuda:0'
    # os.environ['MASTER_ADDR'] = str(config['DDP_MASTER_ADDR'])
    # os.environ['MASTER_PORT'] = str(config['DDP_MASTER_PORT'])
    
    # dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    model, _, _, _, _, _ = network(config)
    model.model.load_state_dict(torch.load(f"{config['model_dir']}/{save_file}", map_location=config['DEVICE']))
    
    # inner_model = torch.load(f"{config['model_dir']}/{save_file}", map_location=config['DEVICE'])
    # if config['METHOD'] == 'ssd':
    #     anchor = anchor_generator(config)
    #     model = ssd_full(config, inner_model, anchor)
    
    model.model = model.model.to(config['DEVICE'])
    model.to(config['DEVICE'])
    
    model.model.eval()
    model.eval()
    
    w_input = config['network']['input_size'][1]
    h_input = config['network']['input_size'][0]
    w_output = 600
    h_output = 600
    
    line_thickness = 1
    text_thickness = 1
    text_scale = 0.5
    
    with np.load('EUA1200.npz') as X:
        K1, d1 = [X[i] for i in ('K', 'd')]

    newcamera1, roi1 = cv2.getOptimalNewCameraMatrix(K1, d1, (640,480), 0)
    
    vfov = 2 * np.arctan(0.5 * 480 / newcamera1[1, 1]) * 180 / np.pi
    camera_height = 2.5
    camera_tilt = 90 - 42.38665372469076
    
    # output
    vc_out = None
    if config['save_video'] is not None:
        vc_out = cv2.VideoWriter(f"{config['model_dir']}/{config['save_video']}", cv2.VideoWriter_fourcc(*'DIVX'), 30, (w_output, h_output))
    
    print(config['DEVICE'])
    # cam
    if config['cam']:
        vc = cv2.VideoCapture(0)
        assert vc.isOpened(), '동영상 파일을 열 수 없습니다.'
        
        while True:
            ret, img = vc.read()
            if not ret:
                continue
            
            img = cv2.undistort(img, K1, d1, None, newcamera1)
            
            start = time.time()
            
            img = cv2.resize(img, (w_input, h_input))
            img_out = cv2.resize(img, (w_output, h_output)).astype(np.uint8)
            img = torch.from_numpy(img).permute([2, 0, 1]).unsqueeze(0).to(config['DEVICE']).float()
            pred = model(img)
            
            for i in range(len(config['network']['task'])):
                if config['network']['task'][i].lower() == 'box2d':
                    detections = pred[0]
                    # detections = [batch, class, score, x1, y1, x2, y2]
                    
                    for detection in detections:
                        if detection[1] > 0.5:
                            label = int(detection[0])
                            score = detection[1]
                            x1 = int(detection[2] * w_output)
                            y1 = int(detection[3] * h_output)
                            x2 = int(detection[4] * w_output)
                            y2 = int(detection[5] * h_output)
                            
                            theta = camera_tilt
                            # object_degree = theta - (y2/h_output) * vfov + vfov / 2
                            object_degree = theta + (1 - y2/h_output) * vfov
                            distance = camera_height * np.tan(object_degree * np.pi / 180)
                            
                            # txt = f"{config['network']['classes'][label]} : {score:.2f}"
                            txt = f"{distance:.2f}m"
                            
                            box_color = (0, 255, 0)
                            cv2.rectangle(img_out, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(img_out, txt, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, box_color, 2)
                            
            end = time.time()
            cv2.putText(img_out, f"FPS : {1/(end-start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if config['show_result']:
                cv2.imshow('result', img_out)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            
            if config['save_video'] is not None:
                vc_out.write(img_out)

    # img
    if config['img_dir'] is not None:
        img_list = os.listdir(config['img_dir'])
        img_list.sort()
        
        for img_file in img_list:
            start = time.time()
            
            try:
                img = cv2.imread(f"{config['img_dir']}/{img_file}")
                img = cv2.resize(img, (w_input, h_input))
                img_out = cv2.resize(img, (w_output, h_output)).astype(np.uint8)
                img = torch.from_numpy(img).permute([2, 0, 1]).unsqueeze(0).to(config['DEVICE']).float()
            except:
                continue
            
            pred = model(img)
            
            for i in range(len(config['network']['task'])):
                if config['network']['task'][i].lower() == 'box2d':
                    detections = pred[0]
                    # detections = [batch, class, score, x1, y1, x2, y2]
                    
                    for detection in detections:
                        if detection[1] > 0.5:
                            label = int(detection[0])
                            score = detection[1]
                            x1 = int(detection[2] * w_output)
                            y1 = int(detection[3] * h_output)
                            x2 = int(detection[4] * w_output)
                            y2 = int(detection[5] * h_output)
                            txt = f"{config['network']['classes'][label]} : {score:.2f}"
                            
                            box_color = colormap[label]
                            cv2.rectangle(img_out, (x1, y1), (x2, y2), box_color, line_thickness)
                            cv2.putText(img_out, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_scale, box_color, text_thickness)
                            
            end = time.time()
            cv2.putText(img_out, f"FPS : {1/(end-start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if config['show_result']:
                cv2.imshow('result', img_out)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                
            if config['save_video'] is not None:
                vc_out.write(img_out)
        
    if config['video'] is not None:
        vc = cv2.VideoCapture(config['video'])
        assert vc.isOpened(), '동영상 파일을 열 수 없습니다.'
        
        while True:
            ret, img = vc.read()
            if not ret:
                vc.release()
                break
            
            start = time.time()
            
            img = cv2.resize(img, (w_input, h_input))
            img_out = cv2.resize(img, (w_output, h_output)).astype(np.uint8)
            img = torch.from_numpy(img).permute([2, 0, 1]).unsqueeze(0).to(config['DEVICE']).float()
            pred = model(img)
            
            for i in range(len(config['network']['task'])):
                if config['network']['task'][i].lower() == 'box2d':
                    detections = pred[0]
                    # detections = [batch, class, score, x1, y1, x2, y2]
                    
                    for detection in detections:
                        if detection[1] > 0.5:
                            label = int(detection[0])
                            score = detection[1]
                            x1 = int(detection[2] * w_output)
                            y1 = int(detection[3] * h_output)
                            x2 = int(detection[4] * w_output)
                            y2 = int(detection[5] * h_output)
                            txt = f"{config['network']['classes'][label]} : {score:.2f}"
                            
                            box_color = colormap[label]
                            cv2.rectangle(img_out, (x1, y1), (x2, y2), box_color, line_thickness)
                            cv2.putText(img_out, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_scale, box_color, text_thickness)
                            
            end = time.time()
            cv2.putText(img_out, f"FPS : {1/(end-start):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if config['show_result']:
                cv2.imshow('result', img_out)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    vc.release()
                    break
                
            if config['save_video'] is not None:
                vc_out.write(img_out)
       
    if config['save_video'] is not None:
        vc_out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    
    # Input
    parser.add_argument('--img_dir', default=None, type=str)
    parser.add_argument('--video', default=None, type=str)
    parser.add_argument('--cam', action='store_true')
    
    # CPU or GPU
    parser.add_argument('--run_cpu', action='store_true')
    
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
    
    # Test
    # opt.model_dir = 'runs/mobilenetv2_ssd_argoseye'
    # opt.model_dir = 'runs/mobilenetv2_ssd_coco2017_woper_1'
    opt.model_dir = 'runs/mobilenetv2_ssd_argococo_crop05_1'
    opt.best = True
    
    opt.cam = True
    # opt.video = 'D:/virtual2.mp4'
    # opt.img_dir = 'C:\\Users\\dqg06\\OneDrive\\Desktop\\TS1'
    
    opt.show_result = True
    # opt.save_video = 'TS1.mp4'
    
    
    

    # read txt to dict
    with open(f"{opt.model_dir}/config.json", 'r') as f:
        config = json.load(f)

    config.update(vars(opt))
    
    # if 'background' in config['network']['classes']:
    #     config['network']['classes'].remove('background')
    #     config['network']['num_classes'] -= 1

    # config['network']['num_classes'] += 1 # background class 추가

    test(config)