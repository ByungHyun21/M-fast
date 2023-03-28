import argparse
import yaml

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from model.network_manager import network_manager
from model.dataloader import get_dataloader
from model.metric import mAP
from model.utils import *
from model.report_manager import report_manager

def train(rank:int, config:dict, manager:report_manager):
    config['DEVICE'] = 'cuda' + ':' + str(rank)    
    os.environ['MASTER_ADDR'] = config['DDP_MASTER_ADDR']
    os.environ['MASTER_PORT'] = config['DDP_MASTER_PORT']
    
    dist.init_process_group(backend=config['DDP_BACKEND'], 
                            rank=rank, 
                            world_size=config['DDP_WORLD_SIZE'], 
                            init_method=config['DDP_INIT_METHOD'])
    
    model, preprocessor, augmentator, loss_func = network_manager(config, rank, istrain=True)
    model.model = DDP(model.model.to(config['DEVICE']), device_ids=[rank], output_device=rank)
    model.to(config['DEVICE'])

    ds_train = get_dataloader(config, 'train', preprocessor=preprocessor, augmentator=augmentator)
    ds_valid = get_dataloader(config, 'valid', preprocessor=preprocessor, augmentator=augmentator)

    # TODO: 옵티마이저를 네트워크마다 다르게 설정할 필요가 있음, Config 파일에 옵티마이저 설정 추가
    lr = config['LR'] * config['DDP_WORLD_SIZE'] * config['BATCH_SIZE_MULTI_GPU'] / 64 # batch size 64
    # optimizer = optim.Adam(model.parameters(), lr=config['LR'], betas=(0.9, 0.999), weight_decay=config['WEIGHT_DECAY'])
    optimizer = optim.SGD(model.model.parameters(), lr=config['LR'], momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
    
    def custom_scheduler(step):
        if step < config['STEPS'][0]:
            lr = 1 # learning_rate = lr0 * lr
        elif step < config['STEPS'][1]:
            lr = 0.1
        else:
            lr = 0.01
        return lr
                
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)
    
    # metric
    if 'mAP' in config['METRIC']:
        metric_mAP = mAP(config)
        metric_mAP.set(config['DATASET'], config['CATEGORY'], config['CLASS'], model.model_class)
    
    step = 0
    epoch = -1
    best_loss = float('inf')
    while True:
        epoch += 1
        if rank == 0:
            print(f"\n\nModel: {config['MODEL']}, epoch: {epoch}, step: {step}")
            
        # # # # #
        # Training
        model.model.train()
        manager.reset()
        
        pbar = tqdm(ds_train, desc='Training', ncols=0) if rank == 0 else ds_train
        for img, gt in pbar:
            step += 1
            
            optimizer.zero_grad()
            pred = model.model(img.to(config['DEVICE']))
            loss = loss_func(pred, gt.to(config['DEVICE']))
            loss[0].backward() # loss[0] = total loss
            optimizer.step()
            scheduler.step()
            
            manager.accumulate_loss(loss)
            if rank == 0:
                pbar.set_postfix_str(manager.loss_print() + f"lr: {scheduler.get_last_lr()[0]:.6f}")
                
            dist.barrier()
            
        dict_train = manager.get_loss_dict('train/')
        manager.wandb_report(rank, epoch, dict_train)

        # # # # #
        # Validation
        model.model.eval()
        manager.reset()
        pbar = tqdm(ds_valid, desc='Validation', ncols=0) if rank == 0 else ds_valid
        for img, gt in pbar:
            pred = model.model(img.to(config['DEVICE']))
            loss = loss_func(pred, gt.to(config['DEVICE']))
            
            manager.accumulate_loss(loss)
            if rank == 0:
                pbar.set_postfix_str(manager.loss_print())
            
            dist.barrier()

        dict_valid = manager.get_loss_dict('valid/')
        manager.wandb_report(rank, epoch, dict_valid)

        # report inference result to wandb
        manager.wandb_report(rank, epoch, {'lr': scheduler.get_last_lr()[0]})
        if ('Object Detection' in config['TASK']) and (epoch % 10 == 0) and (epoch != 0):
            manager.wandb_report_object_detection(rank, epoch, model)
            
        if rank == 0:    
            if 'mAP' in config['METRIC']:
                metric_mAP.reset()
                mAPs = metric_mAP(model)
                manager.wandb_report(epoch, mAPs)
                metric_mAP.print()
            
            # # # # #
            # Save
        
        dist.barrier()
        
        if step >= config['STEPS'][-1]:
            break
        
    dist.destroy_process_group()
        
    if ('mAP' in config['METRIC']) and (rank == 0):
        with open('model/config/dataset/coco.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class)
        metric_mAP.reset()
        mAPs = metric_mAP(model)
        metric_mAP.print()
    
        with open('model/config/dataset/voc.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class)
        metric_mAP.reset()
        mAPs = metric_mAP(model)
        metric_mAP.print()
    
        with open('model/config/dataset/crowdhuman.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class)
        metric_mAP.reset()
        mAPs = metric_mAP(model)
        metric_mAP.print()
    
        with open('model/config/dataset/argoseye.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class)
        metric_mAP.reset()
        mAPs = metric_mAP(model)
        metric_mAP.print()
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/config/ssd/ssd_mobilenet_v2.yaml')
    parser.add_argument('--coco', action='store_true')
    parser.add_argument('--voc', action='store_true')
    parser.add_argument('--crowdhuman', action='store_true')
    parser.add_argument('--argoseye', action='store_true')
    parser.add_argument('--wandb', default=None, type=str)
    opt = parser.parse_args()

    #TODO: 테스트용
    # opt.coco = True
    # opt.wandb = 'byunghyun'

    config = configuration(opt)
    
    manager = report_manager(config)
    
    mp.spawn(train, args=(config, manager), nprocs=torch.cuda.device_count(), join=True)