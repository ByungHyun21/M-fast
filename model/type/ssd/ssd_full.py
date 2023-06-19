import torch
import torch.nn as nn

import numpy as np

class ssd_full(nn.Module):
    def __init__(self, cfg, model, anchor):
        super().__init__()
        self.device = cfg['device']
        self.model_class = cfg['network']['classes']
        self.input_size = cfg['network']['input_size']

        self.model = model
        self.anchor = torch.from_numpy(anchor).float().to(cfg['device'])
        self.activation = nn.Softmax(dim=2)
        
        self.topk = cfg['nms']['topk']
        self.nms_iou_threshold = cfg['nms']['iou_threshold']

    @torch.no_grad()
    def forward(self, x):
        x = self.model(x)
        x = self.postprocess(x)
        return x
    
    def postprocess(self, x):
        """
        x = [batch, anchor_n, class_n+1+4], 4 = [delta_cx, delta_cy, delta_w, delta_h], 1 = background
        self.anchor = [anchor_n, 4], 4 = [cx, cy, w, h]
        """
        class_pred = x[:, :, :-4]
        d_x, d_y, d_w, d_h = torch.split(x[:, :, -4:], [1, 1, 1, 1], dim=2)
        a_x, a_y, a_w, a_h = torch.split(self.anchor, [1, 1, 1, 1], dim=1)
        
        a_x.unsqueeze_(0)
        a_y.unsqueeze_(0)
        a_w.unsqueeze_(0)
        a_h.unsqueeze_(0)
        
        cx = (d_x * a_w / 10.0) + a_x
        cy = (d_y * a_h / 10.0) + a_y
        w = torch.exp(d_w / 5.0) * a_w
        h = torch.exp(d_h / 5.0) * a_h

        x1 = cx - (w / 2.0)
        x2 = cx + (w / 2.0)
        y1 = cy - (h / 2.0)
        y2 = cy + (h / 2.0)
        
        # class_pred = self.activation(class_pred)
        
        score, class_pred = torch.max(class_pred[:, :, 1:], dim=2) # remove background
        box = torch.concat([x1, y1, x2, y2], dim=2).contiguous()
        
        output = self.nms(class_pred, score, box, top_k=self.topk, nms_iou_threshold=self.nms_iou_threshold)

        return output

    def nms(self, cls, score, box, top_k=200, nms_iou_threshold=0.5):
        
        output = []
        
        for i in range(cls.shape[0]):
            c = torch.unsqueeze(cls[i], 1)
            s = torch.unsqueeze(score[i], 1)
            b = box[i]
            detections = torch.concat([c, s, b], dim=1).contiguous()
            
            _, sort_index = torch.sort(s, dim=0, descending=True)
            
            detections = detections[sort_index].squeeze(1)
            # detections = [class, score, x1, y1, x2, y2], sorted by score
            
            keep = torch.zeros((top_k, 6))
            
            idx = 0
            while True:
                if detections.shape[0] == 0:
                    break
                
                if detections[0, 1] < 0.01:
                    break
                
                if idx >= top_k:
                    break
                
                keep[idx] = detections[0]
                
                box1 = detections[0, 2:]
                box2 = detections[1:, 2:]
                
                ious = self.iou(box1, box2)
                iou_idx = (ious < nms_iou_threshold).squeeze(1)
                detections = detections[1:][iou_idx]
                
                
                idx += 1
                
            output.append(keep)
            
        output = torch.stack(output, dim=0)
        return output
                
    def iou(self, box1, box2):
        # box1 = [x1, y1, x2, y2]
        # box2 = n x [x1, y1, x2, y2]
        
        b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, [1, 1, 1, 1], dim=0)
        b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, [1, 1, 1, 1], dim=1)
        
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        x_left = torch.max(b1_x1, b2_x1)
        x_right = torch.min(b1_x2, b2_x2)
        y_top = torch.max(b1_y1, b2_y1)
        y_bottom = torch.min(b1_y2, b2_y2)
        
        common = torch.clamp(x_right - x_left, min=0, max=1) * torch.clamp(y_bottom - y_top, min=0, max=1)
        
        ious = common / (area1 + area2 - common)
        
        return ious 

    def convert_gt(self, gt): 
        # wandb log 같은 곳에서 사용되는 함수
        # Input: preprocessing 된 데이터
        # Output: GT 데이터
        # gt: preprocess 이후의 데이터
        # return: preprocess 이전의 데이터
        conf = gt[:, :, :-4]
        d_x, d_y, d_w, d_h = torch.split(gt[:, :, -4:], [1, 1, 1, 1], dim=2)
        a_x, a_y, a_w, a_h = torch.split(self.anchor, [1, 1, 1, 1], dim=1)
        
        a_x.unsqueeze_(0)
        a_y.unsqueeze_(0)
        a_w.unsqueeze_(0)
        a_h.unsqueeze_(0)
        
        cx = (d_x * a_w / 10.0) + a_x
        cy = (d_y * a_h / 10.0) + a_y
        w = torch.exp(d_w / 5.0) * a_w
        h = torch.exp(d_h / 5.0) * a_h

        x1 = cx - (w / 2.0)
        x2 = cx + (w / 2.0)
        y1 = cy - (h / 2.0)
        y2 = cy + (h / 2.0)
        
        conf = torch.argmax(conf, dim=2) - 1
        
        box = torch.concat([x1, y1, x2, y2], dim=2).contiguous()
        
        original_gt = torch.concat([conf.unsqueeze(2), box], dim=2).contiguous()

        related_anchor = torch.concat([conf.unsqueeze(2), self.anchor.unsqueeze(0)], dim=2).contiguous()

        return original_gt, related_anchor

class ssd_full_onnx(nn.Module):
    # Argoseye에 탑재할 SSD 모델
    def __init__(self, cfg, model, anchor):
        super().__init__()
        self.device = cfg['DEVICE']
        self.model_class = cfg['CLASS']
        self.n_class = len(self.model_class) + 1
        
        self.input_size = cfg['INPUT_SIZE']
        
        self.scale = cfg['anchor']['scales']
        self.grid_w = cfg['anchor']['grid_w']
        self.grid_h = cfg['anchor']['grid_h']
        self.anchor_n = cfg['anchor']['num_anchor']
        
        self.model = model
        self.anchor = torch.from_numpy(anchor).float().to(self.device)
        
        
        # self.activation = nn.Softmax(dim=-1)
        
        self.topk = cfg['nms']['topk']
        self.nms_iou_threshold = cfg['nms']['iou_threshold']

    @torch.no_grad()
    def forward(self, x):
        x = self.model(x)
        x = self.postprocess(x, self.anchor)
        return x
    
    def postprocess(self, x, anchor):
        """
        x = [batch, anchor_n, 4+1+class_n], 4 = [delta_cx, delta_cy, delta_w, delta_h], 1 = background
        self.anchor = [anchor_n, 4], 4 = [cx, cy, w, h]
        """
        class_pred, d_x, d_y, d_w, d_h = torch.split(x.contiguous(), [self.n_class, 1, 1, 1, 1], dim=-1)
        
        a_x, a_y, a_w, a_h = torch.split(anchor.contiguous(), [1, 1, 1, 1], dim=-1)
        
        cx = (d_x * a_w / 10.0) + a_x # Argoseye에서 Cx Cy 계산이 제대로 되지 않음 (왜?)
        cy = (d_y * a_h / 10.0) + a_y
        w = torch.exp(d_w / 5.0) * a_w
        h = torch.exp(d_h / 5.0) * a_h

        x1 = cx - (w / 2.0)
        x2 = cx + (w / 2.0)
        y1 = cy - (h / 2.0)
        y2 = cy + (h / 2.0)
        
        loc = torch.concat([x1, y1, x2, y2], dim=2).contiguous()
        
        output = torch.concat([class_pred, loc], dim=-1).contiguous()
        return output

class ssd_full_argoseye(nn.Module):
    # Argoseye에 탑재할 SSD 모델
    def __init__(self, cfg, model, anchor):
        super().__init__()
        self.device = cfg['device']
        self.model_class = cfg['network']['classes']
        self.n_class = cfg['network']['num_classes']
        
        self.input_size = cfg['network']['input_size']
        
        self.scale = cfg['anchor']['scales']
        self.grid_w = cfg['anchor']['grid_w']
        self.grid_h = cfg['anchor']['grid_h']
        self.anchor_n = cfg['anchor']['num_anchor']
        
        self.model = model
        self.anchor = anchor
        
        
        # self.activation = nn.Softmax(dim=-1)
        
        self.topk = cfg['nms']['topk']
        self.nms_iou_threshold = cfg['nms']['iou_threshold']

    @torch.no_grad()
    def forward(self, x):
        x = self.model(x)
        
        anchor = self.anchor_generator(self.scale, self.grid_w, self.grid_h, self.anchor_n)
        # anchor = torch.from_numpy(self.anchor).float().to(self.device)
        
        x = torch.concat([x, anchor.unsqueeze(0)], dim=2).contiguous()
        # x = self.postprocess(x, anchor)
        return x
    
    def postprocess(self, x, anchor):
        """
        x = [batch, anchor_n, 4+1+class_n], 4 = [delta_cx, delta_cy, delta_w, delta_h], 1 = background
        self.anchor = [anchor_n, 4], 4 = [cx, cy, w, h]
        """
        anchor.unsqueeze_(0)
        
        class_pred, d_x, d_y, d_w, d_h = torch.split(x.contiguous(), [self.n_class, 1, 1, 1, 1], dim=-1)
        
        a_x, a_y, a_w, a_h = torch.split(anchor.contiguous(), [1, 1, 1, 1], dim=-1)
        
        cx = (d_x * a_w / 10.0) + a_x # Argoseye에서 Cx Cy 계산이 제대로 되지 않음 (왜?)
        cy = (d_y * a_h / 10.0) + a_y
        w = torch.exp(d_w / 5.0) * a_w
        h = torch.exp(d_h / 5.0) * a_h

        x1 = cx - (w / 2.0)
        x2 = cx + (w / 2.0)
        y1 = cy - (h / 2.0)
        y2 = cy + (h / 2.0)
        
        loc = torch.concat([x1, y1, x2, y2], dim=2).contiguous()
        
        output = torch.concat([class_pred, loc, d_x, d_y, d_w, d_h, anchor, cx, cy, w, h, x1, y1, x2, y2], dim=-1).contiguous()
        return output
    
    def anchor_generator(self, scale, grid_w, grid_h, anchor_n):
        a1 = self.get_anchor(grid_h[0], grid_w[0], anchor_n[0], scale[0], scale[1])
        a2 = self.get_anchor(grid_h[1], grid_w[1], anchor_n[1], scale[1], scale[2])
        a3 = self.get_anchor(grid_h[2], grid_w[2], anchor_n[2], scale[2], scale[3])
        a4 = self.get_anchor(grid_h[3], grid_w[3], anchor_n[3], scale[3], scale[4])
        a5 = self.get_anchor(grid_h[4], grid_w[4], anchor_n[4], scale[4], scale[5])
        a6 = self.get_anchor(grid_h[5], grid_w[5], anchor_n[5], scale[5], 1.0)
        
        anchor = torch.concat([a1, a2, a3, a4, a5, a6], dim=0).contiguous()
        
        return anchor

    def get_anchor(self, fh, fw, n_anchor, s1, s2):
        grid = torch.linspace(0.0, 1.0, fw+1)
        grid = (grid[:-1] + grid[1:]) / 2.0
        grid_w = torch.tile(torch.unsqueeze(grid, 0), (fh, 1))
        
        grid = torch.linspace(0.0, 1.0, fh+1)
        grid = (grid[:-1] + grid[1:]) / 2.0
        grid_h = torch.tile(torch.unsqueeze(grid, 1), (1, fw))
        
        cx = torch.reshape(grid_w, (-1, 1))
        cy = torch.reshape(grid_h, (-1, 1))
        
        na = fw * fh

        # 1: s1
        aw = s1
        ah = s1
        cw = torch.tile(torch.tensor(aw), (na, 1))
        ch = torch.tile(torch.tensor(ah), (na, 1))
        a1 = torch.concat([cx, cy, cw, ch], dim=1).contiguous()
        
        # 2: sqrt(s1, s2)
        aw = torch.sqrt(torch.tensor(s1 * s2))
        ah = torch.sqrt(torch.tensor(s1 * s2))
        cw = torch.tile(torch.tensor(aw), (na, 1))
        ch = torch.tile(torch.tensor(ah), (na, 1))
        a2 = torch.concat([cx, cy, cw, ch], dim=1).contiguous()
        
        # 3: 2x 1/2x
        # 1.4 1.61 1.82
        aw = s1 * torch.sqrt(torch.tensor(2.0))
        ah = s1 / torch.sqrt(torch.tensor(2.0))
        
        cw = torch.tile(torch.tensor(aw), (na, 1))
        ch = torch.tile(torch.tensor(ah), (na, 1))
        a3_1 = torch.concat([cx, cy, cw, ch], dim=1).contiguous()
        a3_2 = torch.concat([cx, cy, ch, cw], dim=1).contiguous()
        
        if n_anchor != 6:
            anchors = torch.concat([a1, a2, a3_1, a3_2], dim=1).contiguous()
            anchors = torch.reshape(anchors, (-1, 4))
            anchors = torch.clip(anchors, 0.0, 1.0)
            return anchors
        else: 
            # 4: 3x 1/3x
            # 1.7 1.95 2.2
            aw = s1 * torch.sqrt(torch.tensor(3.0))
            ah = s1 / torch.sqrt(torch.tensor(3.0))
            
            cw = torch.tile(torch.tensor(aw), (na, 1))
            ch = torch.tile(torch.tensor(ah), (na, 1))
            a4_1 = torch.concat([cx, cy, cw, ch], dim=1).contiguous()
            a4_2 = torch.concat([cx, cy, ch, cw], dim=1).contiguous()
            anchors = torch.concat([a1, a2, a3_1, a3_2, a4_1, a4_2], dim=1).contiguous()
            anchors = torch.reshape(anchors, (-1, 4))
            anchors = torch.clip(anchors, 0.0, 1.0)
            return anchors