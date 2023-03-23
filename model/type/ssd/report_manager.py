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
        
        self.phase = None # 'T' or 'V' for training and validate

        self.loss_class = AverageMeter()
        self.loss_location = AverageMeter()
        self.loss_all = AverageMeter()
        
        self.iter_start = time.time()
        self.iter_end = time.time()
        
        self.global_step = 0
        self.local_step = 0

        if self.rank == 0 and config['REPORT_WANDB']:
            wandb.init(project='M-FAST', entity='byunghyun', config=config)
        
    def report_wandb(self, rdT, rdV, step):
        if self.rank != 0:
            return 
        
        rdT.update(rdV)
        wandb.log(rdT, step=step)

    def report_model(self):
        if self.rank != 0:
            return 

        # TODO:
        
    def gather(self, l_all, l_cls, l_loc):
        if self.rank != 0:
            return 
        
        self.loss_class.get_data(l_cls.detach().cpu().numpy())
        self.loss_location.get_data(l_loc.detach().cpu().numpy())
        self.loss_all.get_data(l_all.detach().cpu().numpy())
        
        self.global_step += 1
        self.local_step += 1
        
    def report(self):
        if self.rank != 0:
            return 
        
        report_dict = {
            self.phase+'Lall': self.loss_all.get_mean(),
            self.phase+'Lcls': self.loss_class.get_mean(), 
            self.phase+'Lloc': self.loss_location.get_mean()
        }

        return report_dict
        
    def report_print(self):
        if self.rank != 0:
            return 
        
        self.iter_start = time.time()
        if self.phase == 'T':
            string = f'Training | step: {self.local_step}/{self.len_train} | '
        else:
            string = f'Validation | step: {self.local_step}/{self.len_valid} | '
        string += f'{1/(self.iter_start - self.iter_end):.2f} iter/S | '
        string += f'LAll: {self.loss_all.get_mean():.2f} | Lcls: {self.loss_class.get_mean():.2f} | LLoc: {self.loss_location.get_mean():.2f}'
        print(string, end='\r', flush=True)
        self.iter_end = time.time()
        
        sys.stdout.flush()

    def report_reset(self):
        if self.rank != 0:
            return 
        
        self.loss_class.reset()
        self.loss_location.reset()
        self.loss_all.reset()
        
        self.local_step = 0
        
        print("")