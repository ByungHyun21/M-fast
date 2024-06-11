import argparse
import json
import time
import os
import cv2

import numpy as np

from tqdm import tqdm
from model.type.ssd.ssd_full import ssd_full_argoseye
from model.type.ssd.anchor import anchor_generator
from model.network import network

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def convert_onnx(config:dict):
    print('torch version: ', torch.__version__)
    
    # weight load
    files = os.listdir(config['model_dir'])
    save_file = None
    for file in files:
        if config['best'] and file.endswith('best.pth'):
            save_file = file
        if config['last'] and file.endswith('last.pth'):
            save_file = file
    
    assert save_file is not None, 'best나 last를 선택해야 합니다.'
    
    config['device'] = 'cuda:0'
    model, _, _, _, _, _ = network(config)
    model.model.load_state_dict(torch.load(f"{config['model_dir']}/{save_file}", map_location=config['device']))
    
    if config['network']['type'] == 'ssd':
        anchor = anchor_generator(config)
        model = ssd_full_argoseye(config, model.model, anchor)
    
    model.model = model.model.to(config['device'])
    model.to(config['device'])
    
    model.model.eval()
    model.eval()
    
    model(torch.rand(1, 3, 300, 300).to(config['device']))
    
    onnx_file = f"{config['model_dir']}/model.onnx"
    torch.onnx.export(model, torch.rand(1, 3, 300, 300), onnx_file, opset_version=9, export_params=True, verbose=False, do_constant_folding=True)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    
    # Select model weight
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--last', action='store_true')

    opt = parser.parse_args()
    
    # Test
    opt.model_dir = 'runs/mobilenetv2_ssd_argococo_crop05_1'
    opt.best = True    
    
    assert opt.model_dir is not None, 'model_dir을 입력해주세요.'

    # read txt to dict
    with open(f'{opt.model_dir}/config.json', 'r') as f:
        config = json.load(f)

    config.update(vars(opt))

    convert_onnx(config)