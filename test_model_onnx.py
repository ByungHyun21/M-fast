import argparse
import json
import time
import os
import cv2
import numpy as np
import onnxruntime
import random 

random.seed(100)
colormap = []
for i in range(1000):
    colormap.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

def test(config:dict):
    if config['run_cpu']: #cpu
        ep_list = ['CPUExecutionProvider']
    else: #gpu 
        ep_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    ort_session = onnxruntime.InferenceSession(f"{config['model_dir']}/model.onnx", providers=ep_list)
    
    print(f"ep_list : {ep_list}")
    print(f"onnx session : {ort_session.get_providers()}")
    
    w_input = config['network']['input_size'][1]
    h_input = config['network']['input_size'][0]
    w_output = 600
    h_output = 600
    
    line_thickness = 1
    text_thickness = 1
    text_scale = 0.5
    
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
            img_out = cv2.resize(img, (w_output, h_output)).astype(np.uint8)
            img = np.expand_dims(img.transpose(2, 0, 1).astype(np.float32), axis=0)
            
            ort_inputs = {ort_session.get_inputs()[0].name: img}
            detections = ort_session.run(None, ort_inputs)            
            detections = detections[0]
            
            for i in range(len(config['network']['task'])):
                if config['network']['task'][i].lower() == 'box2d':
                    detections = detections[0]
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
                img = np.expand_dims(img.transpose(2, 0, 1).astype(np.float32), axis=0)
            except:
                continue
            
            ort_inputs = {ort_session.get_inputs()[0].name: img}
            detections = ort_session.run(None, ort_inputs)            
            detections = detections[0]
            
            for i in range(len(config['network']['task'])):
                if config['network']['task'][i].lower() == 'box2d':
                    detections = detections[0]
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
            img = np.expand_dims(img.transpose(2, 0, 1).astype(np.float32), axis=0)
            
            ort_inputs = {ort_session.get_inputs()[0].name: img}
            detections = ort_session.run(None, ort_inputs)            
            detections = detections[0]
            
            for i in range(len(config['network']['task'])):
                if config['network']['task'][i].lower() == 'box2d':
                    detections = detections[0]
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
    opt.model_dir = 'runs/mobilenetv2_ssd_argococo_crop05_1'
    opt.best = True
    
    opt.cam = True
    # opt.video = 'D:/construct_02.mp4'
    # opt.img_dir = 'D:/Argoseye/Test/CH2'
    # opt.img_dir = 'sample'
    
    opt.show_result = True
    # opt.save_video = 'test_CH2.mp4'
    
    opt.device = 'gpu'
    
    assert opt.model_dir is not None, '모델 경로를 입력해주세요.'

    # read txt to dict
    with open(f'{opt.model_dir}/config.json', 'r') as f:
        config = json.load(f)

    config.update(vars(opt))

    test(config)