import argparse

import torch

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model.network import network
from model.utils import *

# pip install ptflops
from ptflops import get_model_complexity_info

def train(rank:int, config:dict):
    config['DEVICE'] = 'cuda' + ':' + str(rank)    
    
    model, preprocessor, augmentator, loss_func = network(config, rank, istrain=True)
    

    with torch.cuda.device(0):
        net = model.model.to(config['DEVICE'])
        macs, params = get_model_complexity_info(net, (3, 300, 300), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/config/ssd/mobilenet_v2_voc.yaml')
    parser.add_argument('--coco', action='store_true')
    parser.add_argument('--voc', action='store_true')
    parser.add_argument('--crowdhuman', action='store_true')
    parser.add_argument('--argoseye', action='store_true')
    parser.add_argument('--wandb', default=None, type=str)
    parser.add_argument('--dataset_path', default=None, type=str)
    opt = parser.parse_args()

    opt.coco = True
    assert opt.config is not None, 'config is not defined'

    config = configuration(opt)
    
    mp.spawn(train, args=([config]), nprocs=torch.cuda.device_count(), join=True)