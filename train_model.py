import argparse
import logging
import os
import time

import platform

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import wandb
from model.network_manager import network_manager
from model.dataloader import dataset
from model.metric import mAP
from model.utils import *

# # # # #
# Training
def train(rank:int, config:dict):
    config['DEVICE'] = config['DEVICE'] + ':' + str(rank)
    
    # # # # #
    # Model 
    model, model_full, preprocessor, augmentator, loss, manager = network_manager(config, rank, istrain=True)

    # # # # #
    # with DDP
    dist.init_process_group(config['DDP_BACKEND'], 
                            rank=ddp_rank, 
                            world_size=config['DDP_WORLD_SIZE'], 
                            init_method=config['DDP_INIT_METHOD'])

    model.to(ddp_rank)
    model = DDP(model, device_ids=[ddp_rank], output_device=ddp_rank)

    ds_train = dataset(
        config, ddp_rank, config['DDP_WORLD_SIZE'], config['DEVICE'], 'train', 
        preprocessor=preprocessor, augmentator=augmentator)
    ds_valid = dataset(
        config, ddp_rank, config['DDP_WORLD_SIZE'], config['DEVICE'], 'valid', 
        preprocessor=preprocessor, augmentator=augmentator)

    # # # # # # # #
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['LR'], betas=(0.9, 0.999), weight_decay=config['WEIGHT_DECAY'])
    # optimizer = optim.SGD(model.parameters(), lr=config['LR'], momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
    
    step_per_epoch = config['REPORT_STEP_TRAIN']
    def custom_scheduler(step):
        if step * step_per_epoch < config['STEPS'][0]:
            lr = 1 # learning_rate = lr0 * lr
        elif step * step_per_epoch < config['STEPS'][1]:
            lr = 0.1
        else:
            lr = 0.01
        return lr
                
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)
    # Optimizer
    # # # # # # # #
    
    step = 0
    best_train = float('inf')
    best_valid = float('inf')
    while True:
        if ddp_rank == 0:
            print(f"\n\nproject: {config['PROJECT']}, step: {step}")
            
        # # # # #
        # Training
        dist.barrier()
        model.train()
        manager.report_reset()
        manager.phase = 'T'
        report_step = 0
        
        while True:
            report_step += 1
            step += 1
            
            img, gt = ds_train.batchloader.batch.get()
            optimizer.zero_grad()
            pred = model(img)
            loss_all, loss_cls, loss_loc = loss(pred, gt)
            loss_all.backward()
            optimizer.step()
            
            acc_fg, acc_bg, acc_loc = metric(pred, gt)

            manager.gather(loss_all, loss_cls, loss_loc, acc_fg, acc_bg, acc_loc)
            report_dict_train = manager.report()
            manager.report_print()

            dist.barrier()
            
            if report_step >= config['REPORT_STEP_TRAIN']:
                break

        # # # # #
        # Validation
        dist.barrier()
        model.eval()
        manager.report_reset()
        manager.phase = 'V'
        report_step = 0

        while True:
            report_step += 1
            
            img, gt = ds_valid.batchloader.batch.get()
            pred = model(img)

            loss_all, loss_cls, loss_loc = loss(pred, gt)
            acc_fg, acc_bg, acc_loc = metric(pred, gt)
            
            
            manager.gather(loss_all, loss_cls, loss_loc, acc_fg, acc_bg, acc_loc)
            report_dict_valid = manager.report()
            manager.report_print()

            dist.barrier()
                
            if report_step >= config['REPORT_STEP_VALID']:
                break
            
        scheduler.step()
        dist.barrier()
        
        if config['REPORT_WANDB']:
            manager.report_wandb(report_dict_train, report_dict_valid, step)

        if ddp_rank == 0:
            if not os.path.exists('runs'):
                os.mkdir('runs')
            save_path = 'runs/' + config['PROJECT']
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if best_train > report_dict_train['TLall']:
                best_train = report_dict_train['TLall']
                torch.save(model.module.state_dict(), save_path + '/' + config['PROJECT'] + '_train.pt')

            if best_valid > report_dict_valid['VLall']:
                best_valid = report_dict_valid['VLall']
                torch.save(model.module.state_dict(), save_path + '/' + config['PROJECT'] + '_valid.pt')

        if step >= config['STEPS'][-1]:
            break

    if config['REPORT_EVAL']:
        manager.report_model()

    dist.barrier()
    dist.destroy_process_group()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/config/ssd/ssd_mobilenet_v2.yaml')
    parser.add_argument('--coco', action='store_true')
    parser.add_argument('--voc', action='store_true')
    parser.add_argument('--crowdhuman', action='store_true')
    parser.add_argument('--argoseye', action='store_true')
    opt = parser.parse_args()
    
    config = configuration(opt)

    mp.spawn(train, nprocs=torch.cuda.device_count(), join=True)