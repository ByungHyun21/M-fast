import torch
import torch.nn as nn

class loss(nn.Module):
    def __init__(self):
        super().__init__()
        # # # # #
        # Loss Parameter
        self.alpha = 1.0
        self.SmoothL1Loss = nn.L1Loss(reduction='none')
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, pred, gt):
        n_anchors = gt.shape[1]
        # pred_cls : batch*anchors*class
        # pred_loc : batch*anchors*(cx, cy, w, h)
        pred_cls = pred[0][:, :, :-4]
        pred_loc = pred[0][:, :, -4:]
        
        # gt_cls : batch*anchors*class
        # gt_loc : batch*anchors*(cx, cy, w, h)
        gt_cls = gt[:, :, :-4]
        gt_loc = gt[:, :, -4:]
        
        neg = gt_cls[:, :, 0]
        pos = 1 - neg
        npos = torch.sum(pos, dim=1)

        # # # # #
        # Cross Entropy
        pred_cls = self.softmax(pred_cls)
        entropy = torch.sum(gt_cls * -torch.log(torch.clip(pred_cls, 1e-5, 1.0 - 1e-5)), axis=2)
        
        # # # # #
        # Hard Negative Mining
        e_pos = entropy * pos # positive entropy
        e_neg = entropy * neg # negative entropy
        
        e_neg, _ = e_neg.sort(descending=True)
        _, indices = e_neg.sort(descending=True)
        
        thres = torch.tile((npos * 3).unsqueeze(1), (1, n_anchors))
        
        e_neg = torch.where(indices < thres, e_neg, 0.0)
        
        loss_fg = torch.sum(e_pos, dim=1) 
        loss_bg = torch.sum(e_neg, dim=1)
        
        loss_cls = loss_fg + loss_bg
        # Hard Negative Mining
        # # # # #
        
        # # # # #
        # Regression Loss
        huber_loss = self.SmoothL1Loss(pred_loc, gt_loc)
        huber_loss = torch.mean(huber_loss, dim=2)
        
        loss_loc = self.alpha * torch.sum(pos * huber_loss, axis=1)
        
        # # # # #
        # Loss All
        loss_all = ((loss_cls + loss_loc) / npos).mean()
        
        loss_cls = (loss_cls / npos).mean()
        loss_loc = (loss_loc / npos).mean()
        
        return loss_all, loss_cls, loss_loc