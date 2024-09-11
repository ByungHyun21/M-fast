import argparse
import os
import json
import time

import numpy as np
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.build_pipeline import build_pipeline
from src.dataloader_model import dataloader_model
from src.metric import mAP
from src.report_manager import report_manager

def train(rank:int, cfg:dict):
    cfg['device'] = 'cuda' + ':' + str(rank)    
    os.environ['MASTER_ADDR'] = cfg['DDP']['master_addr']
    os.environ['MASTER_PORT'] = str(cfg['DDP']['master_port'])
    
    dist.init_process_group(backend=cfg['DDP']['backend'], 
                            rank=rank, 
                            world_size=cfg['DDP']['world_size'], 
                            init_method=cfg['DDP']['init_method'])
    
    model, augmentator, loss_func, optimizer, scheduler = build_pipeline(cfg)
    model.model = DDP(model.model.to(cfg['device']), device_ids=[rank], output_device=rank)
    model.to(cfg['device'])
    
    # DataLoader
    batch_size = cfg['batch_size'] // cfg['DDP']['world_size']
    cfg.update({'batch_size_one_gpu':batch_size})
    
    dataset_train = dataloader_model(rank, cfg, 'Train', augmentator=augmentator)
    dataset_valid = dataloader_model(rank, cfg, 'Valid', augmentator=augmentator)
    print(f"dataset_train: {len(dataset_train)}")
    print(f"dataset_valid: {len(dataset_valid)}")
    
    sampler = DistributedSampler(dataset_train) if cfg['DDP']['world_size'] > 1 else None
    ds_train = DataLoader(dataset_train, 
                      batch_size=batch_size,
                      shuffle=True if sampler is None else False, 
                      num_workers=cfg['num_workers'], 
                      pin_memory=True, 
                      drop_last=True, # 간혹 마지막 배치 크기가 1인 경우가 있어서, batch norm에서 문제 발생. 그래서 drop_last=True로 설정 
                      sampler=sampler) # DDP에서만 sampler가 필요함

    ds_valid = DataLoader(dataset_valid, 
                      batch_size=batch_size,
                      shuffle=False, 
                      num_workers=cfg['num_workers'], 
                      pin_memory=True, 
                      drop_last=True, 
                      sampler=None)
    # mAP 잠깐 보류
    # mAP_use = False
    # for idx in range(len(cfg['metric'])):
    #     if 'mAP' == cfg['metric'][idx][0]:
    #         mAP_use = True
    #         mAP_mode = cfg['metric'][idx][1]

    # metric
    # if mAP_use:
    #     mAP_metric = mAP(cfg)
    #     mAP_metric.set(cfg['dataset_root'], cfg['dataset'], model.model_class, mAP_mode)
    
    # save directory
    if rank == 0:
        if not os.path.exists("runs"):
            os.mkdir("runs")
        run_name = f"{cfg['exp_name']}".lower()
        save_dir = f"runs/{run_name}"
        index_dir = 0
        while True:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                break
            else:
                index_dir += 1
                save_dir = f"runs/{run_name}_{index_dir}"

        # save config
        with open(f"{save_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4)

    epoch = -1
    step = 0
    time_start = time.time()
    best_mAP = 0
    # manager = report_manager(cfg, rank)
    while True:
        epoch += 1
        if rank == 0:
            time_current = time.time()
            time_remain = (time_current - time_start) / (epoch+1) * (cfg['end_epoch'] - epoch)
            print(f"\n\nModel: {cfg['exp_name']}, epoch: {epoch}, step: {step}, time: {time_current - time_start:.2f}s, remain: {time_remain:.2f}s")
            print(f"save_dir: {save_dir}")
            
        img_train = []
        gt_train = []
            
        # # # # #
        # Training
        model.eval()
        model.model.train()
        # manager.reset()
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(ds_train, desc='Training', ncols=0) if rank == 0 else ds_train
        for img, gt in pbar:
            pred = model.model(img.to(cfg['device']).contiguous())
            loss = loss_func(pred, gt.to(cfg['device']).contiguous())
            loss[0].backward() # loss[0] = total loss
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            # manager.accumulate_loss(loss)
            # if rank == 0:
            #     pbar.set_postfix_str(manager.loss_print() + f"lr: {scheduler.get_last_lr()[0]:.6f}")

            if len(img_train) < 8:
                img_train.append(img[0].cpu().numpy())
                gt_train.append(gt[0].cpu().numpy())
                
            dist.barrier()
            
        scheduler.step()
            
        # dict_train = manager.get_loss_dict()

        # # # # #
        # Validation
        model.eval()
        model.model.eval()
        # manager.reset()
        pbar = tqdm(ds_valid, desc='Validation', ncols=0) if rank == 0 else ds_valid
        for img, gt in pbar:
            pred = model.model(img.to(cfg['device']).contiguous())
            loss = loss_func(pred, gt.to(cfg['device']).contiguous())
            
            # manager.accumulate_loss(loss)
            # if rank == 0:
            #     pbar.set_postfix_str(manager.loss_print())
            
            dist.barrier()

        # dict_valid = manager.get_loss_dict()
    
        # if epoch % 5 == 0:
        #     if mAP_use:
        #         mAP_metric.reset()
        #         mAPs = mAP_metric(rank, model)
        #         if rank == 0:
        #             mAP_metric.print()
            
        #         # Save Best Model
        #         if rank == 0:
        #             if mAPs['metric/mAP_0.5_0.95'] > best_mAP:
        #                 best_mAP = mAPs['metric/mAP_0.5_0.95']
        #                 torch.save(model.model.module.state_dict(), f"{save_dir}/{run_name}_mAP_best.pth".lower())
        #                 with open(f"{save_dir}/{mAP_metric.dataset_name}_mAP_best.txt", 'w') as f:
        #                     for k, v in mAPs.items():
        #                         f.write(str(k) + ' : '+ str(v) + '\n')

        dist.barrier()
        
        if epoch >= cfg['training']['end_epoch']:
            break

    # if 'mAP' in cfg['METRIC']:
    #     mAP_metric.reset()
    #     mAPs = mAP_metric(rank, model)
    #     if rank == 0:
    #         mAP_metric.print() 

    # Save Last Model
    if rank == 0:
        torch.save(model.model.module.state_dict(), f"{save_dir}/{run_name}_mAP_{best_mAP:.2f}_last.pth".lower())
        # if mAP_use:
        #     with open(f"{save_dir}/mAP_last.txt", 'w') as f:
        #         for k, v in mAPs.items():
        #             f.write(str(k) + ' : '+ str(v) + '\n')
        
    # if mAP_use:
    #     if os.path.exists(f"{cfg['dataset_root']}/'COCO2017'"):
    #         mAP_metric.set(cfg['dataset_root'], 'COCO2017', model.model_class, mAP_mode)
    #         mAP_metric.reset()
    #         mAPs = mAP_metric(rank, model)
    #         mAP_metric.print()
    #         with open(f"{save_dir}/mAP_coco_last.txt", 'w') as f:
    #             for k, v in mAPs.items():
    #                 f.write(str(k) + ' : '+ str(v) + '\n')
    
    #     if os.path.exists(f"{cfg['dataset_root']}/'CrowdHuman'"):
    #         mAP_metric.set(cfg['dataset_root'], 'CrowdHuman', model.model_class, mAP_mode)
    #         mAP_metric.reset()
    #         mAPs = mAP_metric(rank, model)
    #         mAP_metric.print()
    #         with open(f"{save_dir}/mAP_crowdhuman_last.txt", 'w') as f:
    #             for k, v in mAPs.items():
    #                 f.write(str(k) + ' : '+ str(v) + '\n')
    
    #     if os.path.exists(f"{cfg['dataset_root']}/'Argoseye'"):
    #         mAP_metric.set(cfg['dataset_root'], 'Argoseye', model.model_class, mAP_mode)
    #         mAP_metric.reset()
    #         mAPs = mAP_metric(rank, model)
    #         mAP_metric.print()
    #         with open(f"{save_dir}/mAP_argoseye_last.txt", 'w') as f:
    #             for k, v in mAPs.items():
    #                 f.write(str(k) + ' : '+ str(v) + '\n')
        
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_model/monouni/dla34_monouni_rope3d.json')
    parser.add_argument('--dataset_root', type=str, default='D:/Rope3D_cits')
    opt = parser.parse_args()
    
    assert opt.config is not None, 'config is not defined'
    assert opt.dataset_root is not None, 'dataset_root is not defined'
    
    #json read
    with open(opt.config) as f:
        cfg = json.load(f)
    
    cfg['dataset_root'] = opt.dataset_root    
    print('config : ', opt.config)

    #TODO: 테스트용
    cfg['test'] = True
    
    # DDP 에 사용할 포트를 사용하지 않는 포트중 임의로 선택
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    
    cfg['DDP'] = dict()
    cfg['DDP']['backend'] = 'gloo' # 'nccl', 'gloo'
    cfg['DDP']['master_addr'] = 'localhost'
    cfg['DDP']['master_port'] = port
    cfg['DDP']['world_size'] = torch.cuda.device_count()
    cfg['DDP']['init_method'] = 'tcp://localhost:' + str(port)
    
    cfg['exp_name'] = opt.config.split('/')[-1].split('.')[0]
    
    mp.spawn(train, args=([cfg]), nprocs=cfg['DDP']['world_size'], join=True)