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

config = configuration(opt)

assert config['BATCH_SIZE'] % (config['BATCH_PER_GPU'] * config['DDP_WORLD_SIZE']) == 0, 'BATCH_SIZE % (BATCH_PER_GPU * DDP_WORLD_SIZE) != 0'

# # # # #
# Training
def train(rank:int):
    config['DEVICE'] = config['DEVICE'] + ':' + str(rank)
    
    model, preprocessor, augmentator, loss_func, manager = network_manager(config, rank, istrain=True)

    ds_train = dataset(
        config, 0, config['DDP_WORLD_SIZE'], config['DEVICE'], 'train', 
        preprocessor=preprocessor, augmentator=augmentator)
    ds_valid = dataset(
        config, 0, config['DDP_WORLD_SIZE'], config['DEVICE'], 'valid', 
        preprocessor=preprocessor, augmentator=augmentator)

    
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
    metric_mAP = mAP()
    
    #TODO: 테스트용
    with open('model/config/dataset/coco.yaml') as f:
        target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'])
        metric_mAP(model)
    
    step = -1
    epoch = -1
    best_loss = float('inf')
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
            
            img, gt = ds_train.batch.get()
            optimizer.zero_grad()
            pred = model.model(img.to(config['DEVICE']))
            loss = loss_func(pred, gt.to(config['DEVICE']))
            loss[0].backward() # loss[0] = total loss
            optimizer.step()
            scheduler.step()
            
            manager.accumulate_loss(loss)
            pbar.set_postfix_str(manager.loss_print() + f"lr: {scheduler.get_last_lr()[0]:.6f}")
            
            model(img.to(config['DEVICE']))
            
        dict_train = manager.loss_dict('train/')
        manager.wandb_report(epoch, dict_train)


        # # # # #
        # Validation
        model.model.eval()
        manager.reset()
        pbar = tqdm(range(ds_valid.steps_per_epoch), desc='Validation', ncols=0)
        for _ in pbar:
            img, gt = ds_valid.batch.get()
            pred = model.model(img.to(config['DEVICE']))
            loss = loss_func(pred, gt.to(config['DEVICE']))
            
            manager.accumulate_loss(loss)
            pbar.set_postfix_str(manager.loss_print())

        dict_valid = manager.loss_dict('valid/')
        manager.wandb_report(epoch, dict_valid)
        
        # # # # #
        # report sample
        if ('Object Detection' in config['TASK']) and (epoch % 10 == 0):
            manager.wandb_report_sample(epoch)
        
        # # # # #
        # Metric
        if 'mAP' in config['METRIC']:
            metric_mAP.set(config['DATASET'], config['CATEGORY'], config['CLASS'])
            metric_mAP(model)
            pass
        
        # # # # #
        # Save
        
        
        if step >= config['STEPS'][-1]:
            break
        
    if 'mAP' in config['METRIC']:
        with open('model/config/dataset/coco.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'])
        metric_mAP(model)
    
        with open('model/config/dataset/voc.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'])
        metric_mAP(model)
    
        with open('model/config/dataset/crowdhuman.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'])
        metric_mAP(model)
    
        with open('model/config/dataset/argoseye.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'])
        metric_mAP(model)
        
        
        

if __name__ == '__main__':
    train(0)
    
    # mp.spawn(train, nprocs=torch.cuda.device_count(), join=True)