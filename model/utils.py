import os, platform
import yaml
import inspect
from termcolor import colored

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

    # DDP configuration    
    if platform.system() == 'Windows':
        config.update({'DDP_BACKEND': 'gloo'})
    else:
        config.update({'DDP_BACKEND': 'nccl'})
    
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    
    # localhost = one node with multiple GPUs
    config.update({'DDP_INIT_METHOD': 'tcp://localhost:' + str(port)})
    config.update({'DDP_WORLD_SIZE': torch.cuda.device_count()})
        
    return config

def save_model(config, model):
    if not os.path.exists('runs'):
            os.mkdir('runs')
    save_path = 'runs/' + config['MODEL'] + '_' + config['DATASET'] 
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    torch.save(model.module.state_dict(), save_path + '/' + config['PROJECT'] + '_train.pt')
    
def log(string:str):
    file = os.path.basename(__file__)
    func = inspect.currentframe().f_code.co_name
    line = inspect.currentframe().f_lineno
    
    str1 = colored('[' + file + ' ' + func + ':' + str(line) + '] ', 'blue')
    str2 = colored(string, 'green')
    print(str1 + str2)