import torch
import torch.nn as nn

class ssd_metric(nn.Module):
    def __init__(self):
        super().__init__()
        
    def sigmoid(self, x):
        return 1 / (1 +np.exp(-x))
    
    def softmax(self, x):
        f_x = np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)
        return f_x
        
    def forward(self, pred, gt):
        output = pred[0].detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        anchor = pred[1].detach().cpu().numpy()
        
        pred_cls = output[:, :, :-4]
        pred_loc = output[:, :, -4:]
        
        gt_cls = gt[:, :, :-4]
        gt_loc = gt[:, :, -4:]
        
        neg = gt_cls[:, :, 0]
        pos = 1 - neg
        npos = np.sum(pos, axis=1)
        nneg = np.sum(neg, axis=1)
        
        pred_cls = self.softmax(pred_cls)
        acc_cls = np.sum(np.sum(pred_cls * gt_cls, axis=2) * pos, axis=1) / npos
        acc_cls_neg = np.sum(np.sum(pred_cls * gt_cls, axis=2) * neg, axis=1) / nneg
        
        delta_cx, delta_cy, delta_w, delta_h = np.split(pred_loc, 4, axis=2)
        anchor_cx, anchor_cy, anchor_w, anchor_h = np.split(np.expand_dims(anchor, axis=0), 4, axis=2)

        box_cx = (delta_cx * anchor_w / 10.0) + anchor_cx
        box_cy = (delta_cy * anchor_h / 10.0) + anchor_cy
        box_w = np.exp(delta_w / 5.0) * anchor_w
        box_h = np.exp(delta_h / 5.0) * anchor_h
        
        pred_box = np.concatenate([box_cx, box_cy, box_w, box_h], axis=2)

        delta_cx, delta_cy, delta_w, delta_h = np.split(gt_loc, 4, axis=2)
        anchor_cx, anchor_cy, anchor_w, anchor_h = np.split(np.expand_dims(anchor, axis=0), 4, axis=2)

        box_cx = (delta_cx * anchor_w / 10.0) + anchor_cx
        box_cy = (delta_cy * anchor_h / 10.0) + anchor_cy
        box_w = np.exp(delta_w / 5.0) * anchor_w
        box_h = np.exp(delta_h / 5.0) * anchor_h
        
        gt_box = np.concatenate([box_cx, box_cy, box_w, box_h], axis=2)
        
        cx = pred_box[:, :, 0]
        cy = pred_box[:, :, 1]
        w = pred_box[:, :, 2]
        h = pred_box[:, :, 3]
        
        x11 = cx - (w / 2.0)
        x12 = cx + (w / 2.0)
        y11 = cy - (h / 2.0)
        y12 = cy + (h / 2.0)
        
        area1 = np.maximum(x12-x11, 0.0) * np.maximum(y12-y11, 0.0)
        
        cx = gt_box[:, :, 0]
        cy = gt_box[:, :, 1]
        w = gt_box[:, :, 2]
        h = gt_box[:, :, 3]
        
        x21 = cx - (w / 2.0)
        x22 = cx + (w / 2.0)
        y21 = cy - (h / 2.0)
        y22 = cy + (h / 2.0)
        
        area2 = np.maximum(x22-x21, 0.0) * np.maximum(y22-y21, 0.0)
        
        xmin = np.maximum(x11, x21)
        ymin = np.maximum(y11, y21)
        xmax = np.minimum(x12, x22)
        ymax = np.minimum(y12, y22)
        
        common = np.maximum(xmax-xmin, 0.0) * np.maximum(ymax-ymin, 0.0)
        
        ious = common / (area1 + area2 - common)
        
        acc_loc = np.sum(ious * pos, axis=1) / npos
        
        acc_fg = np.mean(acc_cls)
        acc_bg = np.mean(acc_cls_neg)
        acc_loc = np.mean(acc_loc)
        
        return acc_fg, acc_bg, acc_loc