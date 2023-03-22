class ssd_manager():
    def __init__(self, config, ddp_rank):
        self.ddp_rank = ddp_rank
        
        self.phase = None # 'T' or 'V' for training and validate

        self.loss_class = AverageMeter()
        self.loss_location = AverageMeter()
        self.loss_all = AverageMeter()
        self.acc_fg = AverageMeter()
        self.acc_bg = AverageMeter()
        self.acc_location = AverageMeter()
        
        self.iter_start = time.time()
        self.iter_end = time.time()
        
        self.global_step = 0
        self.local_step = 0
        self.len_train = config['REPORT_STEP_TRAIN']
        self.len_valid = config['REPORT_STEP_VALID']

        if ddp_rank == 0 and config['REPORT_WANDB']:
            wandb.init(project='OD_SSD', entity='byunghyun', config=config)
        
    def report_wandb(self, rdT, rdV, step):
        if self.ddp_rank == 0:
            rdT.update(rdV)
            wandb.log(rdT, step=step)

    def report_model(self):

        pass
        
    def gather(self, l_all, l_cls, l_loc, a_fg, a_bg, a_loc):
        if self.ddp_rank == 0:
            self.loss_class.get_data(l_cls.detach().cpu().numpy())
            self.loss_location.get_data(l_loc.detach().cpu().numpy())
            self.loss_all.get_data(l_all.detach().cpu().numpy())
            self.acc_fg.get_data(a_fg)
            self.acc_bg.get_data(a_bg)
            self.acc_location.get_data(a_loc)
            
            self.global_step += 1
            self.local_step += 1
        
    def report(self):
        if self.ddp_rank == 0:
            report_dict = {
                self.phase+'Lall': self.loss_all.get_mean(),
                self.phase+'Lcls': self.loss_class.get_mean(), 
                self.phase+'Lloc': self.loss_location.get_mean(),
                self.phase+'Afg': self.acc_fg.get_mean(),
                self.phase+'Abg': self.acc_bg.get_mean(),
                self.phase+'Aloc': self.acc_location.get_mean(),
            }

            return report_dict
        
    def report_print(self):
        if self.ddp_rank == 0:
            self.iter_start = time.time()
            if self.phase == 'T':
                string = f'Training | step: {self.local_step}/{self.len_train} | '
            else:
                string = f'Validation | step: {self.local_step}/{self.len_valid} | '
            string += f'{1/(self.iter_start - self.iter_end):.2f} iter/S | '
            string += f'LAll: {self.loss_all.get_mean():.2f} | Lcls: {self.loss_class.get_mean():.2f} | LLoc: {self.loss_location.get_mean():.2f} | '
            string += f'Afg: {self.acc_fg.get_mean():.2f} | Abg: {self.acc_bg.get_mean():.2f} | ALoc: {self.acc_location.get_mean():.2f}'
            print(string, end='\r', flush=True)
            self.iter_end = time.time()
            
            sys.stdout.flush()

    def report_reset(self):
        if self.ddp_rank == 0:
            self.loss_class.reset()
            self.loss_location.reset()
            self.loss_all.reset()
            self.acc_fg.reset()
            self.acc_bg.reset()
            self.acc_location.reset()
            
            self.local_step = 0
            
            print("")