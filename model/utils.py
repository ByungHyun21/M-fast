import os, platform
import yaml

import torch

def configuration(opt):
    # Model configuration
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    config.update({'MODEL': config['METHOD'] + '_' + config['TYPE']})
    config.update({'WANDB': opt.wandb})
    
    # Dataset configuration
    assert opt.coco + opt.voc + opt.crowdhuman + opt.argoseye == 1, '\t현재는 단일 데이터셋만 지원'
    
    if opt.coco:
        with open('model/config/dataset/coco.yaml') as f:
            dataset = yaml.load(f, Loader=yaml.SafeLoader)
    if opt.voc:
        with open('model/config/dataset/voc.yaml') as f:
            dataset = yaml.load(f, Loader=yaml.SafeLoader)
    if opt.crowdhuman:
        with open('model/config/dataset/crowdhuman.yaml') as f:
            dataset = yaml.load(f, Loader=yaml.SafeLoader)
    if opt.argoseye:
        with open('model/config/dataset/argoseye.yaml') as f:
            dataset = yaml.load(f, Loader=yaml.SafeLoader)

    config.update(dataset)
    config.update({'DATASET_PATH': opt.dataset_path})
    config.update({'NUM_CLASSES': len(config['CLASS'])})

    # DDP configuration    
    # if platform.system() == 'Windows':
    #     config.update({'DDP_BACKEND': 'gloo'})
    # else:
    #     config.update({'DDP_BACKEND': 'nccl'})
    
    config.update({'DDP_BACKEND': 'gloo'})
    
    # DDP 에 사용할 포트를 사용하지 않는 포트중 임의로 선택
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    
    # localhost = one node with multiple GPUs
    config.update({'DDP_MASTER_ADDR': 'localhost'})
    config.update({'DDP_MASTER_PORT': str(port)})
    config.update({'DDP_INIT_METHOD': 'tcp://localhost:' + str(port)})
    config.update({'DDP_WORLD_SIZE': torch.cuda.device_count()})
        
    return config