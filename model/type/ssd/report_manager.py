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
    
    def wandb_report(self, epoch,  dict_out):
        if self.rank == 0 and (config['WANDB'] is not None):
            wandb.log(dict_out, step=epoch)
            
    def wandb_report_sample(self, epoch):
        # # this is the order in which my classes will be displayed
        # display_ids = {"car" : 0, "truck" : 1, "person" : 2, "traffic light" : 3, "stop sign" : 4,
        #             "bus" : 5, "bicycle": 6, "motorbike" : 7, "parking meter" : 8, "bench": 9,
        #             "fire hydrant" : 10, "aeroplane" : 11, "boat" : 12, "train": 13}
        # # this is a revese map of the integer class id to the string class label
        # class_id_to_label = { int(v) : k for k, v in display_ids.items()}


        # def bounding_boxes(filename, v_boxes, v_labels, v_scores, log_width, log_height):
        #     # load raw input photo
        #     raw_image = load_img(filename, target_size=(log_height, log_width))
        #     all_boxes = []
        #     # plot each bounding box for this image
        #     for b_i, box in enumerate(v_boxes):
        #         # get coordinates and labels
        #         box_data = {"position" : {
        #         "minX" : box.xmin,
        #         "maxX" : box.xmax,
        #         "minY" : box.ymin,
        #         "maxY" : box.ymax},
        #         "class_id" : display_ids[v_labels[b_i]],
        #         # optionally caption each box with its class and score
        #         "box_caption" : "%s (%.3f)" % (v_labels[b_i], v_scores[b_i]),
        #         "domain" : "pixel",
        #         "scores" : { "score" : v_scores[b_i] }}
        #         all_boxes.append(box_data)


        #     # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        #     box_image = wandb.Image(raw_image, boxes = {"predictions": {"box_data": all_boxes, "class_labels" : class_id_to_label}})
        #     return box_image
        
        pass
