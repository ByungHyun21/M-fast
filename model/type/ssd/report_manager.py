import wandb
import time, sys, os
import cv2

import torch

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
        self.wandb_entity = config['WANDB']

        self.loss_name = config['LOSS']
        n_loss = len(config['LOSS'])
        self.loss = []
        for i in range(n_loss):
            self.loss.append(AverageMeter())
        
        if self.rank == 0 and (self.wandb_entity is not None):
            wandb.init(project='M-FAST', entity=self.wandb_entity, config=config)
        
    def reset(self):
        for i in range(len(self.loss)):
            self.loss[i].reset()
            
    def accumulate_loss(self, loss):
        for i in range(len(self.loss)):
            self.loss[i].get_data(loss[i].item())

    def loss_print(self):
        str_out = ''
        
        for i in range(len(self.loss)):
            str_out += f"{self.loss_name[i]}: {self.loss[i].get_mean():.2f}, "
        return str_out
    
    def get_loss_dict(self, prefix):
        dict_out = {}
        for i in range(len(self.loss)):
            dict_out[prefix + self.loss_name[i]] = self.loss[i].get_mean()
        return dict_out
    
    def wandb_report(self, epoch,  dict_out):
        if self.rank == 0 and (self.wandb_entity is not None):
            wandb.log(dict_out, step=epoch)
            
    def wandb_report_object_detection(self, epoch, model):
        sample_dirs = os.listdir('sample')
        
        class_id_to_label = {}
        for i in range(len(model.model_class)):
            class_id_to_label.update({i:model.model_class[i]})
            
        model.eval()
        for sample_dir in sample_dirs:
            img_list = os.listdir(os.path.join('sample', sample_dir))
            
            for img_name in img_list:
                img = cv2.imread(os.path.join('sample', sample_dir, img_name))
                h, w, _ = img.shape
                detections = model(torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(model.device))
                # detections : [batch, [class, score, xmin, ymin, xmax, ymax]]
                
                all_boxes = []
                for detection in detections:
                    box_data = {"position" : {
                    "minX" : detection[2] * w,
                    "maxX" : detection[4] * w,
                    "minY" : detection[3] * h,
                    "maxY" : detection[5] * h} ,
                    "class_id" : model.model_class[detection[0]],
                    # optionally caption each box with its class and score
                    "box_caption" : "%s (%.3f)" % (detection[0], detection[1]),
                    "domain" : "pixel",
                    "scores" : { "score" : detection[1] }}
                    all_boxes.append(box_data)

                # log to wandb: raw image, predictions, and dictionary of class labels for each class id
                if self.rank == 0 and (self.wandb_entity is not None):
                    wandb.Image(img, boxes = {"predictions": {"box_data": all_boxes, "class_labels" : class_id_to_label}})
                