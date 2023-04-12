import argparse
import yaml

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tqdm import tqdm
from model.network import network
from model.dataloader import dataset
from model.metric import mAP
from model.utils import *
from model.report_manager import report_manager

def train(rank:int, config:dict):
    config['DEVICE'] = 'cuda' + ':' + str(rank)    
    os.environ['MASTER_ADDR'] = config['DDP_MASTER_ADDR']
    os.environ['MASTER_PORT'] = config['DDP_MASTER_PORT']
    
    dist.init_process_group(backend=config['DDP_BACKEND'], 
                            rank=rank, 
                            world_size=config['DDP_WORLD_SIZE'], 
                            init_method=config['DDP_INIT_METHOD'])
    
    model, preprocessor, augmentator, loss_func = network(config, rank, istrain=True)
    model.model = DDP(model.model.to(config['DEVICE']), device_ids=[rank], output_device=rank)
    model.to(config['DEVICE'])

    # DataLoader
    batch_size = config['BATCH_SIZE_MULTI_GPU'] // config['DDP_WORLD_SIZE']
    config.update({'BATCH_SIZE':batch_size})
    
    dataset_train = dataset(config, 'train', preprocessor=preprocessor, augmentator=augmentator)
    dataset_valid = dataset(config, 'valid', preprocessor=preprocessor, augmentator=augmentator)
    
    sampler = DistributedSampler(dataset_train) if config['DDP_WORLD_SIZE'] > 1 else None
    ds_train = DataLoader(dataset_train, 
                      batch_size=batch_size,
                      shuffle=True if sampler is None else False, 
                      num_workers=config['WORKERS'], 
                      pin_memory=True, 
                      drop_last=False, 
                      sampler=sampler) # Sampler will shuffle dataset in DDP Training

    ds_valid = DataLoader(dataset_valid, 
                      batch_size=batch_size,
                      shuffle=False, 
                      num_workers=config['WORKERS'], 
                      pin_memory=True, 
                      drop_last=False, 
                      sampler=None)

    # TODO: 옵티마이저를 네트워크마다 다르게 설정할 필요가 있음, Config 파일에 옵티마이저 설정 추가
    lr0 = config['LR'] * config['BATCH_SIZE_MULTI_GPU'] / 32 # batch size 64

    biases = list()
    not_biases = list()
    for param_name, param in model.model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

    # optimizer = optim.Adam(model.parameters(), lr=lr0, betas=(0.9, 0.999), weight_decay=config['WEIGHT_DECAY'])
    optimizer = optim.SGD(model.model.parameters(), lr=lr0, momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
    # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr0}, {'params': not_biases}],
    #                             lr=lr0, momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
    optimizer.zero_grad()
    
    def custom_scheduler(step):
        if step < config['STEPLR'][0]:
            lr = 1 # learning_rate = lr0 * lr
        elif step < config['STEPLR'][1]:
            lr = 0.1
        else:
            lr = 0.01
        return lr
                
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)
    
    # metric
    if 'mAP' in config['METRIC']:
        metric_mAP = mAP(config)
        metric_mAP.set(config['DATASET'], config['CATEGORY'], config['CLASS'], model.model_class)
    
    epoch = -1
    step = 0
    best_loss = float('inf')
    manager = report_manager(config, rank)
    while True:
        epoch += 1
        if rank == 0:
            print(f"\n\nModel: {config['MODEL']}, epoch: {epoch}")
            
        img_train = []
        gt_train = []
            
        # # # # #
        # Training
        model.eval()
        model.model.train()
        manager.reset()
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(ds_train, desc='Training', ncols=0) if rank == 0 else ds_train
        for img, gt in pbar:
            pred = model.model(img.to(config['DEVICE']).contiguous())
            loss = loss_func(pred, gt.to(config['DEVICE']).contiguous())
            loss[0].backward() # loss[0] = total loss
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
            step += 1
            manager.accumulate_loss(loss)
            if rank == 0:
                pbar.set_postfix_str(manager.loss_print() + f"lr: {scheduler.get_last_lr()[0]:.6f}")

            if len(img_train) < 8:
                img_train.append(img[0].cpu().numpy())
                gt_train.append(gt[0].cpu().numpy())
                
            dist.barrier()
            
        dict_train = manager.get_loss_dict('train/')
        manager.wandb_report(step, dict_train)

        # # # # #
        # Validation
        model.eval()
        model.model.eval()
        manager.reset()
        pbar = tqdm(ds_valid, desc='Validation', ncols=0) if rank == 0 else ds_valid
        for img, gt in pbar:
            pred = model.model(img.to(config['DEVICE']).contiguous())
            loss = loss_func(pred, gt.to(config['DEVICE']).contiguous())
            
            manager.accumulate_loss(loss)
            if rank == 0:
                pbar.set_postfix_str(manager.loss_print())
            
            dist.barrier()

        dict_valid = manager.get_loss_dict('valid/')
        manager.wandb_report(step, dict_valid)

        # report inference result to wandb
        manager.wandb_report(step, {'lr': scheduler.get_last_lr()[0]})
        if ('Object Detection' in config['TASK']) and (epoch % 10 == 0):
            manager.wandb_report_object_detection(step, model)
            manager.wandb_report_object_detection_training(step, model, img_train, gt_train)
            
        if epoch % 5 == 0:
            if 'mAP' in config['METRIC']:
                metric_mAP.reset()
                mAPs = metric_mAP(rank, model)
                if rank == 0:
                    manager.wandb_report(step, mAPs)
                    metric_mAP.print()
            
            # # # # #
            # Save
        
        dist.barrier()
        
        if step >= config['STEPLR'][-1]:
            break

    if 'mAP' in config['METRIC']:
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        if rank == 0:
            manager.wandb_report(step, mAPs)
            metric_mAP.print() 
               
    dist.destroy_process_group()
        
    if 'mAP' in config['METRIC']:
        with open('model/config/dataset/coco.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class)
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        metric_mAP.print()
    
        with open('model/config/dataset/voc.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class)
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        metric_mAP.print()
    
        with open('model/config/dataset/crowdhuman.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class)
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        metric_mAP.print()
    
        with open('model/config/dataset/argoseye.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class)
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        metric_mAP.print()
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/config/ssd/vgg16_voc.yaml')
    parser.add_argument('--coco', action='store_true')
    parser.add_argument('--voc', action='store_true')
    parser.add_argument('--crowdhuman', action='store_true')
    parser.add_argument('--argoseye', action='store_true')
    parser.add_argument('--wandb', default=None, type=str)
    parser.add_argument('--dataset_path', default=None, type=str)
    opt = parser.parse_args()

    #TODO: 테스트용
    # opt.voc = True
    # opt.wandb = 'byunghyun'
    # opt.dataset_path = 'C:\dataset'
    # opt.dataset_path = 'C:\\Users\\dqg06\\Desktop'
    
    assert opt.config is not None, 'config is not defined'
    assert opt.coco or opt.voc or opt.crowdhuman or opt.argoseye, 'dataset is not defined'
    assert opt.dataset_path is not None, 'dataset path is not defined'

    config = configuration(opt)
    
    mp.spawn(train, args=([config]), nprocs=torch.cuda.device_count(), join=True)