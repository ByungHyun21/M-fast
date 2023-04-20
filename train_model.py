import argparse
import yaml
import time

import numpy as np
import torch

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
# model.model = torch.load('runs/ssd_vgg16_voc0712/ssd_vgg16_voc0712_map_best.pth', map_location=config['DEVICE'])
def train(rank:int, config:dict):
    config['DEVICE'] = 'cuda' + ':' + str(rank)    
    os.environ['MASTER_ADDR'] = config['DDP_MASTER_ADDR']
    os.environ['MASTER_PORT'] = config['DDP_MASTER_PORT']
    
    dist.init_process_group(backend=config['DDP_BACKEND'], 
                            rank=rank, 
                            world_size=config['DDP_WORLD_SIZE'], 
                            init_method=config['DDP_INIT_METHOD'])
    
    model, preprocessor, augmentator, loss_func, optimizer, scheduler = network(config, rank, istrain=True)
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
                      drop_last=True, # 간혹 마지막 배치 크기가 1인 경우가 있어서, batch norm에서 문제 발생. 그래서 drop_last=True로 설정 
                      sampler=sampler) # DDP에서만 sampler가 필요함

    ds_valid = DataLoader(dataset_valid, 
                      batch_size=batch_size,
                      shuffle=False, 
                      num_workers=config['WORKERS'], 
                      pin_memory=True, 
                      drop_last=True, 
                      sampler=None)

    # metric
    if 'mAP' in config['METRIC']:
        metric_mAP = mAP(config)
        metric_mAP.set(config['DATASET'], config['CATEGORY'], config['CLASS'], model.model_class, config['MAP_MODE'])
    
    # save directory
    if rank == 0:
        if not os.path.exists("runs"):
            os.mkdir("runs")
        run_name = f"{config['METHOD']}_{config['TYPE']}_{config['DATASET']}".lower()
        save_dir = f"runs/{run_name}"
        index_dir = 0
        while True:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                break
            else:
                index_dir += 1
                save_dir = f"runs/{run_name}_{index_dir}"

        with open(f"{save_dir}/configuration.txt", 'w') as f:
            for k, v in config.items():
                f.write(str(k) + ' : '+ str(v) + '\n')

    epoch = -1
    step = 0
    time_start = time.time()
    best_mAP = 0
    manager = report_manager(config, rank)
    while True:
        epoch += 1
        if rank == 0:
            time_current = time.time()
            time_remain = (time_current - time_start) / (step+1) * (config['STEPLR'][-1] - step)
            print(f"\n\nModel: {config['MODEL']}, epoch: {epoch}, step: {step}, time: {time_current - time_start:.2f}s, remain: {time_remain:.2f}s")
            
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
            
                # Save Best Model
                if rank == 0:
                    if mAPs['metric/mAP_0.5_0.95'] > best_mAP:
                        best_mAP = mAPs['metric/mAP_0.5_0.95']
                        torch.save(model.model, f"{save_dir}/{run_name}_mAP_best.pth".lower())
                        with open(f"{save_dir}/mAP_best.txt", 'w') as f:
                            for k, v in mAPs.items():
                                f.write(str(k) + ' : '+ str(v) + '\n')

        dist.barrier()
        
        if step >= config['STEPLR'][-1]:
            break

    if 'mAP' in config['METRIC']:
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        if rank == 0:
            manager.wandb_report(step, mAPs)
            metric_mAP.print() 

    # Save Last Model
    if rank == 0:
        if 'mAP' in config['METRIC']:
            torch.save(model.model, f"{save_dir}/{run_name}_mAP_{best_mAP:.2f}_last.pth".lower())
            with open(f"{save_dir}/mAP_last.txt", 'w') as f:
                for k, v in mAPs.items():
                    f.write(str(k) + ' : '+ str(v) + '\n')
        
    if 'mAP' in config['METRIC']:
        with open('model/config/dataset/coco.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class, config['MAP_MODE'])
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        metric_mAP.print()
        with open(f"{save_dir}/mAP_coco_last.txt", 'w') as f:
            for k, v in mAPs.items():
                f.write(str(k) + ' : '+ str(v) + '\n')
    
        with open('model/config/dataset/voc.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class, config['MAP_MODE'])
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        metric_mAP.print()
        with open(f"{save_dir}/mAP_voc_last.txt", 'w') as f:
            for k, v in mAPs.items():
                f.write(str(k) + ' : '+ str(v) + '\n')
    
        with open('model/config/dataset/crowdhuman.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class, config['MAP_MODE'])
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        metric_mAP.print()
        with open(f"{save_dir}/mAP_crowdhuman_last.txt", 'w') as f:
            for k, v in mAPs.items():
                f.write(str(k) + ' : '+ str(v) + '\n')
    
        with open('model/config/dataset/argoseye.yaml') as f:
            target = yaml.load(f, Loader=yaml.SafeLoader)
        metric_mAP.set(target['DATASET'], target['CATEGORY'], target['CLASS'], model.model_class, config['MAP_MODE'])
        metric_mAP.reset()
        mAPs = metric_mAP(rank, model)
        metric_mAP.print()
        with open(f"{save_dir}/mAP_argoseye_last.txt", 'w') as f:
            for k, v in mAPs.items():
                f.write(str(k) + ' : '+ str(v) + '\n')
        
    dist.destroy_process_group()

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

    #TODO: 테스트용
    # opt.voc = True
    opt.argoseye = True
    opt.wandb = 'byunghyun'
    # opt.dataset_path = 'C:\dataset'
    opt.dataset_path = 'C:\\Users\\dqg06\\OneDrive\\Desktop'
    
    assert opt.config is not None, 'config is not defined'
    assert opt.coco or opt.voc or opt.crowdhuman or opt.argoseye, 'dataset is not defined'
    assert opt.dataset_path is not None, 'dataset path is not defined'

    config = configuration(opt)
    
    mp.spawn(train, args=([config]), nprocs=torch.cuda.device_count(), join=True)