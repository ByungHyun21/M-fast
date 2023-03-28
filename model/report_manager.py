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
    def __init__(self, config):
        self.wandb_entity = config['WANDB']

        self.loss_name = config['LOSS']
        n_loss = len(config['LOSS'])
        self.loss = []
        for i in range(n_loss):
            self.loss.append(AverageMeter())
        
        self.wandb_enabled = False
        if self.wandb_entity is not None:
            wandb.init(project='M-FAST', entity=self.wandb_entity, config=config)
            self.wandb_enabled = True
        
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
    
    def wandb_report(self, rank, epoch,  dict_out):
        if rank != 0:
            return
        
        if self.wandb_entity is not None:
            wandb.log(dict_out, step=epoch)
            
    def wandb_report_object_detection(self, rank, epoch, model):
        if rank != 0:
            return 
        
        sample_dirs = os.listdir('sample')
        
        class_id_to_label = {}
        for i in range(len(model.model_class)):
            class_id_to_label.update({int(i):model.model_class[i]})
            
        model.eval()
        for sample_dir in sample_dirs:
            img_list = os.listdir(os.path.join('sample', sample_dir))
            
            wandb_images = []
            for img_name in img_list:
                img = cv2.imread(os.path.join('sample', sample_dir, img_name))
                img_resize = cv2.resize(img, (model.input_size[1], model.input_size[0]))
                
                h, w, _ = img.shape
                detections = model(torch.from_numpy(img_resize).permute(2, 0, 1).float().unsqueeze(0).to(model.device))
                # detections : [batch, N_pred, [class, score, xmin, ymin, xmax, ymax]]
                detections = detections[0].cpu().detach().numpy()
                # detections : N_pred, [class, score, xmin, ymin, xmax, ymax]
                
                all_boxes = []
                for detection in detections:
                    box_data = {"position" : {
                    "minX" : float(detection[2] * w),
                    "maxX" : float(detection[4] * w),
                    "minY" : float(detection[3] * h),
                    "maxY" : float(detection[5] * h)} ,
                    "class_id" : int(detection[0]),
                    # optionally caption each box with its class and score
                    "box_caption" : "%s (%.3f)" % (str(int(detection[0])), float(detection[1])),
                    "domain" : "pixel",
                    "scores" : { "score" : float(detection[1]) }}
                    all_boxes.append(box_data)

                # log to wandb: raw image, predictions, and dictionary of class labels for each class id
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                wandb_images.append(wandb.Image(img, boxes = {"predictions": {"box_data": all_boxes, "class_labels" : class_id_to_label}}))
                
            if self.wandb_entity is not None:    
                wandb.log({"Object Detection/" + sample_dir: wandb_images}, step=epoch)
                