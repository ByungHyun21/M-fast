import wandb
import time, sys

class AverageMeter:
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        
    def get_data(self, data):
        self.cnt += 1
        self.sum += data
    
    def get_mean(self):
        return self.sum / self.cnt

    def reset(self):
        self.cnt = 0
        self.sum = 0

class report_manager():
    def __init__(self, config, rank):
        self.rank = rank

        self.loss_name = config['LOSS']
        n_loss = len(config['LOSS'])
        self.loss = []
        for i in range(n_loss):
            self.loss.append(AverageMeter())
        
        self.step = 0

        if self.rank == 0 and (config['WANDB'] is not None):
            wandb.init(project='M-FAST', entity=config['WANDB'], config=config)
        
    def reset(self):
        self.step = 0
        for i in range(len(self.loss)):
            self.loss[i].reset()
            
    def accumulate_loss(self, loss):
        for i in range(len(self.loss)):
            self.loss[i].get_data(loss[i].item())
            
        self.step += 1
        
    def loss_print(self):
        str_out = ''
        
        for i in range(len(self.loss)):
            str_out += f"{self.loss_name[i]}: {self.loss[i].get_mean():.2f}, "
        return str_out
    
    def loss_dict(self, prefix):
        dict_out = {}
        for i in range(len(self.loss)):
            dict_out[prefix + self.loss_name[i]] = self.loss[i].get_mean()
        return dict_out
    
    