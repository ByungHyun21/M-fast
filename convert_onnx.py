import argparse
import yaml
import time
import os
import cv2

import numpy as np

from tqdm import tqdm
from model.type.ssd.ssd_full import ssd_full_argoseye
from model.type.ssd.anchor import anchor_generator
from model.utils import *

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def convert_onnx(config:dict):
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
        model = ssd_full_argoseye(config, inner_model, anchor)
    
    # model.model = model.model.to(config['DEVICE'])
    # model.to(config['DEVICE'])
    
    model.model.eval()
    model.eval()
    
    onnx_file = f"{config['model_dir']}/model.onnx"
    torch.onnx.export(model, torch.rand(1, 3, 300, 300), onnx_file, opset_version=10, export_params=True)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='runs/ssd_mobilenet_v2_argoseye_decay')

    # Save Video (저장할 비디오 이름름)
    parser.add_argument('--save_video', default=None, type=str)
    
    # Select model weight
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--last', action='store_true')

    opt = parser.parse_args()
    
    # Test
    opt.best = True    

    # read txt to dict
    with open(f"{opt.model_dir}/configuration.txt", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.update(vars(opt))

    convert_onnx(config)