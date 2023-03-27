import torch
import torch.nn as nn

class loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # # # # #
        # Loss Parameter
        self.alpha = config['LOSS_ALPHA']
        self.SmoothL1Loss = nn.L1Loss(reduction='none')
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, pred, gt):
        # pred : batch*anchors*(4+1+class)
        # gt : batch*anchors*(4+1+class)
        cls = pred[:, :, 4:]
        loc = pred[:, :, :4]
        
        cls_gt = gt[:, :, 4:]
        loc_gt = gt[:, :, :4]
        
        neg = cls_gt[:, :, 0]
        pos = 1 - neg
        npos = torch.sum(pos, dim=1)

        # # # # #
        # Cross Entropy
        # cls_softmax = self.softmax(cls)
        # entropy = torch.sum(cls_gt * -torch.log(torch.clip(cls_softmax, 1e-7, 1.0 - 1e-7)), axis=2)
        cls_sigmoid = self.sigmoid(cls)
        entropy = torch.mean(cls_gt * -torch.log(torch.clip(cls_sigmoid, 1e-7, 1.0 - 1e-7)), axis=2) + \
            torch.mean((1 - cls_gt) * -torch.log(torch.clip(1 - cls_sigmoid, 1e-7, 1.0 - 1e-7)), axis=2)
        
        # # # # #
        # Hard Negative Mining
        e_pos = entropy * pos # positive entropy
        e_neg = entropy * neg # negative entropy
        
        e_neg, _ = e_neg.sort(descending=True)
        _, indices = e_neg.sort(descending=True)
        
        thres = torch.tile((npos * 3).unsqueeze(1), (1, pred.shape[1]))
        
        e_neg = torch.where(indices < thres, e_neg, 0.0)
        
        loss_fg = torch.sum(e_pos, dim=1) 
        loss_bg = torch.sum(e_neg, dim=1)
        
        loss_cls = loss_fg + loss_bg
        # Hard Negative Mining
        # # # # #
        
        # # # # #
        # Regression Loss
        huber_loss = self.SmoothL1Loss(loc, loc_gt)
        huber_loss = torch.mean(huber_loss, dim=2)
        
        loss_loc = self.alpha * torch.sum(pos * huber_loss, dim=1)
        
        # # # # #
        # Loss All
        loss_all = ((loss_cls + loss_loc) / npos).mean()
        
        loss_cls = (loss_cls / npos).mean()
        loss_loc = (loss_loc / npos).mean()
        
        return loss_all, loss_cls, loss_loc