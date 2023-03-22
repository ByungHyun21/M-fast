import argparse
import logging
import os
import time
import yaml
import platform
import cv2

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import wandb

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.od_dataloader import od_dataset
from model.network_manager import network_manager

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/mv2_test.yaml', help='config has training info like epoch, batch_size, ... etc')
parser.add_argument('--gpu', type=str, default='0', help='GPUs eg. 0 or 0,1 or 2,3,4,5')
opt = parser.parse_args()
with open(opt.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# # # # #
# torch, CUDA, gpu Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    device = 'cpu'

ddp_size = torch.cuda.device_count()

"""
multi-gpu 에서 학습은 아직 고려되지 않음. -> od_dataloader 수정해야함 batchloader 같은거 
"""

# # # # #
# Training
def train(ddp_rank):
    
    # # # # #
    # Model 
    model, preprocess, augmentator, loss, metric, manager = network_manager(config, ddp_rank)

    # # # # #
    # with DDP
    os.environ['MASTER_ADDR'] = config['MASTER_ADDR']
    os.environ['MASTER_PORT'] = config['MASTER_PORT']
    dist.init_process_group("gloo", rank=ddp_rank, world_size=ddp_size)

    model.to(ddp_rank)
    if config['PRETRAINED'] == 'torchvision':
        model = DDP(model, device_ids=[ddp_rank], output_device=ddp_rank, find_unused_parameters=True)
    else:
        model = DDP(model, device_ids=[ddp_rank], output_device=ddp_rank)

    ds_train = od_dataset(
        config, ddp_rank, ddp_size, device, 'train', 
        preprocessor=preprocess, augmentator=augmentator)
    ds_valid = od_dataset(
        config, ddp_rank, ddp_size, device, 'valid', 
        preprocessor=preprocess, augmentator=augmentator)

    # # # # # # # #
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['LR'], betas=(0.9, 0.999), weight_decay=0.0005)
    
    step_per_epoch = ds_train.n_data // (config['BATCH'] / ddp_size)
    def custom_scheduler(epoch):
        if epoch * step_per_epoch < config['STEPS'][0]:
            lr = 1 # learning_rate = lr0 * lr
        elif epoch * step_per_epoch < config['STEPS'][1]:
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
            print(f"\nproject: {config['PROJECT']}, step: {step}\n")
            
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
            
            pred = model(img)
            
            for i in range(10):
                cls = pred[0][i, :, :2]
                loc = pred[0][i, :, 2:]
                anchor = pred[1]
                
                cls_gt = gt[i, :, :2]
                loc_gt = gt[i, :, 2:]
                
                d_x, d_y, d_w, d_h = torch.split(loc, [1, 1, 1, 1], dim=1)
                a_x, a_y, a_w, a_h = torch.split(anchor, [1, 1, 1, 1], dim=1)
                
                cx = (d_x * a_w / 10.0) + a_x
                cy = (d_y * a_h / 10.0) + a_y
                w = torch.exp(d_w / 5.0) * a_w
                h = torch.exp(d_h / 5.0) * a_h

                x1 = cx - (w / 2.0)
                x2 = cx + (w / 2.0)
                y1 = cy - (h / 2.0)
                y2 = cy + (h / 2.0)

                pred_box = torch.concat([cls, x1, y1, x2, y2], dim=1).detach().cpu().numpy()
                
                d_x, d_y, d_w, d_h = torch.split(loc_gt, [1, 1, 1, 1], dim=1)
                a_x, a_y, a_w, a_h = torch.split(anchor, [1, 1, 1, 1], dim=1)
                
                cx = (d_x * a_w / 10.0) + a_x
                cy = (d_y * a_h / 10.0) + a_y
                w = torch.exp(d_w / 5.0) * a_w
                h = torch.exp(d_h / 5.0) * a_h

                x1 = cx - (w / 2.0)
                x2 = cx + (w / 2.0)
                y1 = cy - (h / 2.0)
                y2 = cy + (h / 2.0)

                gt_box = torch.concat([cls_gt, x1, y1, x2, y2], dim=1).detach().cpu().numpy()
                
                im_show = img[i].permute((1, 2, 0)).detach().cpu().numpy().astype(np.uint8).copy()
                h, w, c = im_show.shape
                
                for idx, box in enumerate(gt_box):
                    if box[0] == 1:
                        continue
                    
                    x1 = int(box[-4] * w)
                    y1 = int(box[-3] * h)
                    x2 = int(box[-2] * w)
                    y2 = int(box[-1] * h)
                    
                    color = (0, 255, 0)
                    cv2.rectangle(im_show, (x1, y1), (x2, y2), color, 1)
                    
                    box = anchor[idx]
                    
                    x1 = int((box[-4] - box[-2]/2.0) * w)
                    y1 = int((box[-3] - box[-1]/2.0) * h)
                    x2 = int((box[-4] + box[-2]/2.0) * w)
                    y2 = int((box[-3] + box[-1]/2.0) * h)
                    
                    color = (0, 0, 255)
                    cv2.rectangle(im_show, (x1, y1), (x2, y2), color, 1)

                cv2.imwrite('debug/img_with_gt' + str(i) + '.png', im_show)
            
        return

    


if __name__ == '__main__':
    print('nprocs: ', torch.cuda.device_count())
    mp.spawn(train, nprocs=torch.cuda.device_count(), join=True)