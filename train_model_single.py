import argparse

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
opt = parser.parse_args()

#TODO: 테스트용
opt.coco = True

config = configuration(opt)

# # # # #
# Training
def train(rank:int):
    config['DEVICE'] = config['DEVICE'] + ':' + str(rank)
    
    # # # # #
    # Model 
    model, model_full, preprocessor, augmentator, loss_func, manager = network_manager(config, rank, istrain=True)

    ds_train = dataset(
        config, 0, config['DDP_WORLD_SIZE'], config['DEVICE'], 'train', 
        preprocessor=preprocessor, augmentator=augmentator)
    ds_valid = dataset(
        config, 0, config['DDP_WORLD_SIZE'], config['DEVICE'], 'valid', 
        preprocessor=preprocessor, augmentator=augmentator)

    # # # # # # # #
    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=config['LR'], betas=(0.9, 0.999), weight_decay=config['WEIGHT_DECAY'])
    optimizer = optim.SGD(model.parameters(), lr=config['LR'], momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
    
    def custom_scheduler(step):
        if step < config['STEPS'][0]:
            lr = 1 # learning_rate = lr0 * lr
        elif step < config['STEPS'][1]:
            lr = 0.1
        else:
            lr = 0.01
        return lr
                
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)
    # Optimizer
    # # # # # # # #
    
    step = -1
    epoch = -1
    best_loss = float('inf')
    while True:
        epoch += 1
        print(f"\n\nModel: {config['MODEL']}, epoch: {epoch}, step: {step}")
            
        # # # # #
        # Training
        model.train()
        for _ in range(config['STEPS_PER_EPOCH'][0]):
            img, gt = ds_train.batch.get()
            optimizer.zero_grad()
            pred = model(img.to('cuda'))
            loss = loss_func(pred, gt.to('cuda'))
            loss[0].backward()
            optimizer.step()
            
            step += 1
            scheduler.step()

        # # # # #
        # Validation
        model.eval()
        for _ in range(config['STEPS_PER_EPOCH'][1]):
            img, gt = ds_valid.batch.get()
            pred = model(img.to('cuda'))
            loss = loss_func(pred, gt.to('cuda'))

        if step >= config['STEPS'][-1]:
            break

if __name__ == '__main__':
    train(0)
    
    # mp.spawn(train, nprocs=torch.cuda.device_count(), join=True)