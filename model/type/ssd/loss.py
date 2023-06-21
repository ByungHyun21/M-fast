import torch
import torch.nn as nn

class loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # # # # #
        # Loss Parameter
        self.alpha_class = cfg['loss']['alpha']['class']
        self.alpha_location = cfg['loss']['alpha']['location']
        
        
        # self.SmoothL1Loss = nn.L1Loss(reduction='mean')
        self.SmoothL1Loss = nn.HuberLoss(reduction='mean', delta=1.0)
        # self.softmax = nn.Softmax(dim=2)
        # self.sigmoid = nn.Sigmoid()
        # self.cross_entropy = nn.CrossEntropyLoss(reduce=False, reduction='none')
        
    def forward(self, pred, gt):
        # pred : batch*anchors*(4+1+class)
        # gt : batch*anchors*(4+1+class)
        cls_pred = pred[:, :, :-4].contiguous()
        loc_pred = pred[:, :, -4:].contiguous()
        
        cls_gt = gt[:, :, :-4].contiguous()
        loc_gt = gt[:, :, -4:].contiguous()
        
        background = cls_gt[:, :, 0].contiguous()
        
        neg_idx = background == 1
        pos_idx = background == 0
        
        npos = torch.sum(pos_idx, dim=1)
        n_hard_neg = npos * 3

        # Softmax Cross Entropy
        # cls_softmax = self.softmax(cls_pred)
        # entropy = torch.sum(cls_gt * -torch.log(torch.clip(cls_softmax, 1e-7, 1.0 - 1e-7)), axis=2)
        
        # entropy = torch.sum(cls_gt * -torch.log(torch.clip(cls_pred, 1e-7, 1.0 - 1e-7)), axis=2)
        
        #sigmoid Cross Entropy
        entropy = torch.sum(cls_gt * -torch.log(torch.clip(cls_pred, 1e-7, 1.0 - 1e-7)), axis=2) + \
            torch.sum((1 - cls_gt) * -torch.log(torch.clip(1 - cls_pred, 1e-7, 1.0 - 1e-7)), axis=2)
        
        # Positive Loss
        loss_positive = entropy[pos_idx]

        # Hard Negative Mining (Negative Loss)
        entropy_neg = entropy.clone()
        entropy_neg[pos_idx] = 0.0
        
        entropy_neg_desending, _ = entropy_neg.sort(dim=1, descending=True)
        _, indices = entropy_neg_desending.sort(dim=1, descending=True)
        
        hard_neg_indices = indices < n_hard_neg.unsqueeze(1)
        
        loss_negative = entropy_neg_desending[hard_neg_indices]
        
        loss_conf = (loss_positive.sum() + loss_negative.sum()) / npos.sum()
        
        # Regression Loss
        loss_loc = self.SmoothL1Loss(loc_pred[pos_idx], loc_gt[pos_idx])
        
        # # # # #
        # Loss All
        loss_all = self.alpha_class * loss_conf + self.alpha_location * loss_loc
        
        return loss_all, loss_conf, loss_loc
    
    
if __name__ == "__main__":
    # Unit Test
    import numpy as np
    config = {
        'LOSS_ALPHA': 1.0
    }
    
    loss = loss(config)
    
    pred = np.array([
        [
        [1.0, 1.0, 1.0, 1.0, 6, -6, -6],
        [1.0, 1.0, 1.0, 1.0, 6, -6, -6],
        ], 
        [
        [1.0, 1.0, 1.0, 1.0, 6, -6, -6],
        [1.0, 1.0, 1.0, 1.0, 6, -6, -6],
        ]
    ])
    gt = np.array([
        [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        ],
        [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        ]
    ])
    
    loss_all, loss_cls, loss_loc = loss(
        torch.from_numpy(pred).float(), torch.from_numpy(gt).float())
    print("Loss All : ", loss_all)
    print("Loss Cls : ", loss_cls)
    print("Loss Loc : ", loss_loc)
