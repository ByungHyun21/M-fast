import argparse
import yaml

import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm
from model.network_manager import network_manager
from model.dataloader import dataset
from model.metric import mAP
from model.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='model/config/ssd/ssd_mobilenet_v2.yaml')
parser.add_argument('--coco', action='store_true')
parser.add_argument('--voc', action='store_true')
parser.add_argument('--crowdhuman', action='store_true')
parser.add_argument('--argoseye', action='store_true')
parser.add_argument('--wandb', default=None, type=str)
opt = parser.parse_args()

#TODO: 테스트용
opt.coco = True
opt.wandb = 'byunghyun'

config = configuration(opt)

assert config['ACCUMULATE_BATCH_SIZE'] % (config['BATCH_PER_GPU'] * config['DDP_WORLD_SIZE']) == 0, 'ACCUMULATE_BATCH_SIZE % (BATCH_PER_GPU * DDP_WORLD_SIZE) != 0'

# # # # #
# Training
def train(rank:int):
    # GPU configuration
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        config['DEVICE'] = 'cuda' + ':' + str(rank)
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        config['DEVICE'] = 'cpu'
    
    model, preprocessor, augmentator, loss_func, manager = network_manager(config, rank, istrain=True)
    model.model.to(config['DEVICE'])
    model.to(config['DEVICE'])

    ds_train = dataset(
        config, 0, config['DDP_WORLD_SIZE'], config['DEVICE'], 'train', 
        preprocessor=preprocessor, augmentator=augmentator)
    ds_valid = dataset(
        config, 0, config['DDP_WORLD_SIZE'], config['DEVICE'], 'valid', 
        preprocessor=preprocessor, augmentator=augmentator)

    # TODO: 옵티마이저를 네트워크마다 다르게 설정할 필요가 있음, Config 파일에 옵티마이저 설정 추가
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
    scaler = torch.cuda.amp.GradScaler()
    while True:
        epoch += 1
        print(f"\n\nModel: {config['MODEL']}, epoch: {epoch}, step: {step}")
            
        # # # # #
        # Training
        model.model.train()
        manager.reset()
        pbar = tqdm(range(ds_train.steps_per_epoch), desc='Training', ncols=0)
        for _ in pbar:
            step += 1
            
            optimizer.zero_grad()
            for i in range(ds_train.iters_per_step):
                img, gt = ds_train.buffer_batch.get()
                pred = model.model(img.to(config['DEVICE']))
                loss = loss_func(pred, gt.to(config['DEVICE']))
                
                loss_acc = loss[0] / ds_train.iters_per_step
                
                scaler.scale(loss_acc).backward()
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            manager.accumulate_loss(loss)
            pbar.set_postfix_str(manager.loss_print() + f"lr: {scheduler.get_last_lr()[0]:.6f}")
            
            # loss[0].backward() # loss[0] = total loss
            # optimizer.step()
            # scheduler.step()
            
        dict_train = manager.get_loss_dict('train/')
        manager.wandb_report(epoch, dict_train)

        # # # # #
        # Validation
        model.model.eval()
        manager.reset()
        pbar = tqdm(range(ds_valid.steps_per_epoch), desc='Validation', ncols=0)
        for _ in pbar:
            img, gt = ds_valid.buffer_batch.get()
            pred = model.model(img.to(config['DEVICE']))
            loss = loss_func(pred, gt.to(config['DEVICE']))
            
            manager.accumulate_loss(loss)
            pbar.set_postfix_str(manager.loss_print())

        dict_valid = manager.get_loss_dict('valid/')
        manager.wandb_report(epoch, dict_valid)
        
        # # # # #
        # report etc
        manager.wandb_report(epoch, {'lr': scheduler.get_last_lr()[0]})
        if ('Object Detection' in config['TASK']) and (epoch % 10 == 0) and (epoch != 0):
            manager.wandb_report_object_detection(epoch, model)
        
        # # # # #
        # Metric
        if 'mAP' in config['METRIC']:
            metric_mAP.reset()
            mAPs = metric_mAP(model)
            manager.wandb_report(epoch, mAPs)
            metric_mAP.print()
        
        # # # # #
        # Save
        
        
        if step >= config['STEPS'][-1]:
            break
        
    if 'mAP' in config['METRIC']:
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
    train(0)
    
    # mp.spawn(train, nprocs=torch.cuda.device_count(), join=True)