import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm

class mAP(object):
    def __init__(self, cfg):
        super().__init__()
        self.input_size = cfg['network']['input_size']
        self.test = cfg['test']
        self.model_type = cfg['network']['type']
        
        
    def set(self, dataset_root, dataset_name, model_classes, mode='101point'):
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.model_classes = model_classes
        self.mode = mode
        
        self.data = []
        self.common_classes = []
        sub_dirs = os.listdir(f"{self.dataset_root}/{self.dataset_name}/Valid/Label")
        for sub_dir in sub_dirs:
            json_list = os.listdir(f"{self.dataset_root}/{self.dataset_name}/Valid/Label/{sub_dir}")
            
            for json_file in json_list:
                json_path = f"{self.dataset_root}/{self.dataset_name}/Valid/Label/{sub_dir}/{json_file}"
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                isinterest = False
                objects = json_data['object']
                for obj in objects:
                    # check box2d exist
                    if 'box2d' in obj and obj['class'] in self.model_classes:
                        isinterest = True
                        if obj['class'] not in self.common_classes:
                            self.common_classes.append(obj['class'])
                            
                if isinterest:
                    image_name = json_path.replace('Label', 'Image').replace('json', 'jpg')
                    self.data.append({'image': image_name, 'label': json_path})
                
                if self.test and len(self.data) > 100:
                    break

        self.model2mAPClass = []
        for model_class in self.model_classes:
            if model_class in self.common_classes:
                self.model2mAPClass.append(self.common_classes.index(model_class))
            else:
                self.model2mAPClass.append(-1)
                
        
            

        #mAP Calculation
        # mAP 0.5:0.05:0.95
        # mAP 0.5
        # mAP 0.75
        # mAP small (object area < (1/6)^2))
        # mAP medium (object area > (1/6)^2) and (object area < (1/3)^2)
        # mAP large (object area > (1/3)^2))
        self.calculators = []
        self.calculators.append(mAP_calculator(0, 1, 0.5, self.common_classes, self.model2mAPClass, mode=self.mode))                   # mAP 0.5
        self.calculators.append(mAP_calculator(0, 1, 0.55, self.common_classes, self.model2mAPClass, mode=self.mode))                  # mAP 0.55
        self.calculators.append(mAP_calculator(0, 1, 0.6, self.common_classes, self.model2mAPClass, mode=self.mode))                   # mAP 0.6
        self.calculators.append(mAP_calculator(0, 1, 0.65, self.common_classes, self.model2mAPClass, mode=self.mode))                  # mAP 0.65
        self.calculators.append(mAP_calculator(0, 1, 0.7, self.common_classes, self.model2mAPClass, mode=self.mode))                   # mAP 0.7
        self.calculators.append(mAP_calculator(0, 1, 0.75, self.common_classes, self.model2mAPClass, mode=self.mode))                  # mAP 0.75
        self.calculators.append(mAP_calculator(0, 1, 0.8, self.common_classes, self.model2mAPClass, mode=self.mode))                   # mAP 0.8
        self.calculators.append(mAP_calculator(0, 1, 0.85, self.common_classes, self.model2mAPClass, mode=self.mode))                  # mAP 0.85
        self.calculators.append(mAP_calculator(0, 1, 0.9, self.common_classes, self.model2mAPClass, mode=self.mode))                   # mAP 0.9
        self.calculators.append(mAP_calculator(0, 1, 0.95, self.common_classes, self.model2mAPClass, mode=self.mode))                  # mAP 0.95
        self.calculators.append(mAP_calculator(0, (1/6)**2, 0.5, self.common_classes, self.model2mAPClass, mode=self.mode))            # mAP small
        self.calculators.append(mAP_calculator((1/6)**2, (1/3)**2, 0.6, self.common_classes, self.model2mAPClass, mode=self.mode))     # mAP medium
        self.calculators.append(mAP_calculator((1/3)**2, 1, 0.7, self.common_classes, self.model2mAPClass, mode=self.mode))            # mAP large
        
        self.mAP = {}
        
    def reset(self):
        for calculator in self.calculators:
            calculator.reset()
        
        self.mAP = {}
        
    def __call__(self, rank, model):
        model.eval()
        model.model.eval()
        
        pbar = tqdm(self.data, desc=self.dataset_name + ' : calculation TP/FP Table', ncols=0) if rank == 0 else self.data
        for d in pbar: 
            # mAP caclulation one by one
            img, label, box = self.read_data(d)
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
            
            pred = model(torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(model.device))
            
            for calculator in self.calculators:
                calculator.get_data(pred, label, box)
        
        pbar = tqdm(self.calculators, desc=self.dataset_name + ' : calculation mAP', ncols=0) if rank == 0 else self.calculators
        for calculator in pbar:
            calculator.calc_mAP()
        
        # Precision/Recall Table
        name = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.mAP.update({'metric/mAP_0.5_0.95': 0})
        for idx, calculator in enumerate(self.calculators[:10]):
            self.mAP.update({'metric/mAP_' + str(name[idx]): calculator.get_mAP_mean()})
            self.mAP['metric/mAP_0.5_0.95'] += name[idx] * self.mAP['metric/mAP_' + str(name[idx])]
        
        self.mAP['metric/mAP_0.5_0.95'] /= sum(name)
        
        name = ['small', 'medium', 'large']
        for idx, calculator in enumerate(self.calculators[10:]):
            self.mAP.update({'metric/mAP_' + str(name[idx]): calculator.get_mAP_mean()})
            
        return self.mAP
        
    def read_data(self, data):
        label = []
        box = []
        
        json_file = data['label']
        
        with open(json_file) as f:
            json_data = json.load(f)
            
        for obj in json_data['object']:
            l = obj['class']
            
            if l not in self.common_classes:
                continue
            
            if 'box2d' not in obj:
                continue
            
            cx = obj['box2d']['cx']
            cy = obj['box2d']['cy']
            w = obj['box2d']['w']
            h = obj['box2d']['h']

            #cx, cy, w, h -> x1, y1, x2, y2
            x1 = cx - w/2
            x2 = cx + w/2
            y1 = cy - h/2
            y2 = cy + h/2
            
            label.append(self.model2mAPClass[self.model_classes.index(l)])
            box.append([x1, y1, x2, y2])
            
        img = cv2.imread(data['image'])
            
        return img, label, box
    
    def print(self):
        print('dataset :', self.dataset_name)
        print('mAP 0.5_0.95 :', self.mAP['metric/mAP_0.5_0.95'])
        print('mAP 0.5 :', self.mAP['metric/mAP_0.5'])
        print('mAP 0.75 :', self.mAP['metric/mAP_0.75'])
        print('mAP small :', self.mAP['metric/mAP_small'])
        print('mAP medium :', self.mAP['metric/mAP_medium'])
        print('mAP large :', self.mAP['metric/mAP_large'])
        
class mAP_calculator(object):
    def __init__(self, min_area, max_area, iou_threshold, common_class, model2mAPClass, mode='all'):
        super().__init__()
        self.min_area = min_area
        self.max_area = max_area
        self.iou_threshold = iou_threshold
        self.n_class = len(common_class)
        self.common_class = common_class
        self.model2mAPClass = model2mAPClass
        
        self.mode = mode

        self.reset()
        
    def reset(self):
        self.n_gt = np.zeros(self.n_class)
        self.table_TPFP = []
        self.mAP = []
        for _ in range(self.n_class):
            self.table_TPFP.append([])
            self.mAP.append(0)
        
        self.mAP_mean = 0
            
    def get_data(self, pred, label, box):
        # pred : [B, N_pred, [class, score, x1, y1, x2, y2]]
        # label : [B, N]
        # box : [B, N, [x1, y1, x2, y2]]
        
        pred = pred[0].cpu().numpy()
        
        pt = self.pred2table(pred)
        dt = self.dataset2table(label, box, self.min_area, self.max_area)
        # pt : [N_class, pred] pred: [class, score, x1, y1, x2, y2]
        # dt : [N_class, gt] dataset: [class, score, x1, y1, x2, y2]
        
        # calculate [score, TP/FP]
        for idx, (p, d) in enumerate(zip(pt, dt)):
            self.n_gt[idx] += len(d) # accumulate n_gt
            
            if len(p) == 0:
                continue
            
            if len(d) == 0: # no gt
                table_TPFP = []
                for p_ in p:
                    table_TPFP.append([p_[1], 0]) # All prediction is FP
            else:
                p = np.array(p)
                d = np.array(d)
                table_TPFP = self.calc_TPFP(p, d, self.iou_threshold)
                
            for m in table_TPFP:
                self.table_TPFP[idx].append(m)
    
    def get_mAP(self):
        return self.mAP
    
    def get_mAP_mean(self):
        return self.mAP_mean
    
    def pred2table(self, pred):
        # pred : [B, N_pred, [class, score, x1, y1, x2, y2]]
        # return : [N_class, N_pred]
        
        table = []
        for _ in range(self.n_class):
            table.append([])
        
        for p in pred:
            idx = self.model2mAPClass[int(p[0])]
            if idx == -1:
                continue
            table[idx].append(p)
        
        return table
    
    def dataset2table(self, label, box, min_area, max_area):
        # label : [N]
        # box : [N, [x1, y1, x2, y2]]
        # return : [N_class, N]
        
        table = []
        for _ in range(self.n_class):
            table.append([])
            
        for l, b in zip(label, box):
            area = (b[2] - b[0]) * (b[3] - b[1])
            if (area < min_area) or (area > max_area):
                continue
            
            table[l].append([l, 1.0, b[0], b[1], b[2], b[3]]) # score = 1.0
        
        return table
    
    def calc_TPFP(self, p, d, iou_threshold):
        
        ious = self.jaccard_iou_boxes(p[:, 2:], d[:, 2:])
        # ious : [N_pred, N_gt]
        
        table_TPFP = []
        
        n_pred = ious.shape[0]
        n_gt = ious.shape[1]
        matched = np.zeros((n_pred, 1))
        for i in range(n_gt):
            max_iou_col = np.max(ious[:, i])
            if max_iou_col < iou_threshold: # no match, iou < iou_threshold
                continue
            
            max_idx = np.argmax(ious[:, i])
            matched[max_idx] = 1
            ious[max_idx, :] = 0
            
        for i in range(n_pred):
            if matched[i] == 1:
                table_TPFP.append([p[i][1], 1])
            else:
                table_TPFP.append([p[i][1], 0])
                
        return table_TPFP
    
    def jaccard_iou_boxes(self, boxes1, boxes2):
        # boxes1 : [N, 4], [x1, y1, x2, y2]
        # boxes2 : [M, 4], [x1, y1, x2, y2]
        # return : [N, M], iou
        
        N = boxes1.shape[0]
        M = boxes2.shape[0]
        
        ious = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                ious[i, j] = self.jaccard_iou_box(boxes1[i], boxes2[j])
        
        return ious
    
    def jaccard_iou_box(self, box1, box2):
        x1, y1, x2, y2 = 0, 1, 2, 3
        xmin = max(box1[x1], box2[x1])
        xmax = min(box1[x2], box2[x2])
        ymin = max(box1[y1], box2[y1])
        ymax = min(box1[y2], box2[y2])
        
        common_area = max(xmax - xmin, 0) * max(ymax - ymin, 0)
        
        area1 = (box1[x2] - box1[x1]) * (box1[y2] - box1[y1])
        area2 = (box2[x2] - box2[x1]) * (box2[y2] - box2[y1])
        
        iou = common_area / (area1 + area2 - common_area)
        
        return iou
    
    def calc_mAP(self):
        
        pr_curve = []
        for _ in range(self.n_class):
            pr_curve.append([])
        
        # table_TPFP to pr_curve
        for idx, t in enumerate(self.table_TPFP):
            if len(t) == 0:
                continue
            if self.n_gt[idx] == 0:
                continue
            
            t = np.array(t)
            order = np.argsort(-t[:, 0]) # sort by descending order
            t = t[order] # sort by score
            
            sum_TP = 0
            sum_FP = 0
            
            for i in range(len(t)):
                if t[i][1] == 1:
                    sum_TP += 1
                else:
                    sum_FP += 1
            
                # pr_curve : n_class * [recall, precision]    
                pr_curve[idx].append([sum_TP / self.n_gt[idx], sum_TP / (sum_TP + sum_FP)])

        # pr_curve to mAP
        for idx, pr_curve_one_class in enumerate(pr_curve):
            if len(pr_curve_one_class) == 0:
                self.mAP[idx] = 0
                continue
            
            pr_point = [[0, 1]] # [recall, precision]
            for pr in pr_curve_one_class:
                if pr[1] >= pr_point[-1][1]: # precision is higher than previous point
                    pr_point[-1][0] = pr[0]
                    pr_point[-1][1] = pr[1]
                else: # precision is lower than previous point
                    if pr_point[-1][0] == pr[0]:
                        continue
                    else:
                        pr_point.append(pr)
            
            if self.mode == 'all':
                for i in range(len(pr_point[1:])):
                    # mAP : sum of (recall - previous_recall) * precision
                    self.mAP[idx] += (pr_point[i+1][0] - pr_point[i][0]) * pr_point[i+1][1]
            elif self.mode == '11point': # Pascal VOC mAP
                for i in range(11):
                    recall_min = i / 10
                    precision = 0
                    for pr in pr_point[1:]:
                        if pr[0] >= recall_min:
                            precision = max(precision, pr[1])
                        
                    self.mAP[idx] += precision / 11
            elif self.mode == '101point': # COCO mAP
                for i in range(101):
                    recall_min = i / 100
                    precision = 0
                    for pr in pr_point[1:]:
                        if pr[0] >= recall_min:
                            precision = max(precision, pr[1])
                        
                    self.mAP[idx] += precision / 101

        
        count = 0
        for mAP in self.mAP:
            if mAP != 0:
                count += 1
                self.mAP_mean += mAP
                
        if count != 0:
            self.mAP_mean /= count
            self.mAP_mean *= 100.0
        else:
            self.mAP_mean = 0.0