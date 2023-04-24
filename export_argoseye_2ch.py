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
    
    config['DEVICE'] = 'cuda:0'
    os.environ['MASTER_ADDR'] = str(config['DDP_MASTER_ADDR'])
    os.environ['MASTER_PORT'] = str(config['DDP_MASTER_PORT'])
    
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    model, _, _, _, _, _ = network(config)
    model.model.load_state_dict(torch.load(f"{config['model_dir']}/{save_file}", map_location=config['DEVICE']))
    # inner_model = torch.load(f"{config['model_dir']}/{save_file}", map_location=config['DEVICE'])
    if config['METHOD'] == 'ssd':
        model = ssd_full_argoseye(config, model.model)
    
    model.model = model.model.to(config['DEVICE'])
    model.to(config['DEVICE'])
    
    model.model.eval()
    model.eval()
    
    model(torch.rand(1, 3, 300, 300).to(config['DEVICE']))
    
    onnx_file = f"{config['model_dir']}/model.onnx"
    # torch.onnx.export(model, torch.rand(1, 3, 300, 300), onnx_file, opset_version=10, export_params=True, verbose=True, do_constant_folding=True)
    torch.onnx.export(model, torch.rand(1, 3, 300, 300), onnx_file, opset_version=10, export_params=True, verbose=False, do_constant_folding=True)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    
    # Select model weight
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--last', action='store_true')

    opt = parser.parse_args()
    
    # Test
    opt.model_dir = 'runs/ssd_mobilenet_v2_argoseye'
    opt.best = True    
    
    assert opt.model_dir is not None, 'model_dir을 입력해주세요.'

    # read txt to dict
    with open(f"{opt.model_dir}/configuration.txt", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config.update(vars(opt))

    convert_onnx(config)