import torch
import torch.nn as nn

class ssd_full(nn.Module):
    def __init__(self, config, model, anchor):
        super().__init__()
        self.device = config['DEVICE']
        self.model_class = config['CLASS']
        self.input_size = config['INPUT_SIZE']

        self.model = model
        self.anchor = torch.from_numpy(anchor).float().to('cpu').to(config['DEVICE'])
        # self.activation = nn.Sigmoid()
        self.activation = nn.Softmax(dim=2)
        
        self.topk = config['TOPK']
        self.nms_iou_threshold = config['NMS_IOU_THRESHOLD']

    @torch.no_grad()
    def forward(self, x):
        x = self.model(x)
        x = self.postprocess(x)
        return x
    
    def postprocess(self, x):
        """
        x = [batch, anchor_n, 4+1+class_n], 4 = [delta_cx, delta_cy, delta_w, delta_h], 1 = background
        self.anchor = [anchor_n, 4], 4 = [cx, cy, w, h]
        """
        class_pred = x[:, :, 4:]
        d_x, d_y, d_w, d_h = torch.split(x[:, :, :4], [1, 1, 1, 1], dim=2)
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
        
        class_pred = self.activation(class_pred)
        
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
        # gt: preprocess 이후의 데이터
        # return: preprocess 이전의 데이터
        conf = gt[:, :, 4:]
        d_x, d_y, d_w, d_h = torch.split(gt[:, :, :4], [1, 1, 1, 1], dim=2)
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

    def postprocess_openvino(self, x):
        """
        x = [batch, anchor_n, 4+1+class_n], 4 = [delta_cx, delta_cy, delta_w, delta_h], 1 = background
        self.anchor = [anchor_n, 4], 4 = [cx, cy, w, h]
        """
        cls = x[:, :, 4:]
        d_x, d_y, d_w, d_h = torch.split(x[:, :, :4], [1, 1, 1, 1], dim=2)
        a_x, a_y, a_w, a_h = torch.split(self.anchor, [1, 1, 1, 1], dim=1)
        
        cx = (d_x * a_w / 10.0) + a_x
        cy = (d_y * a_h / 10.0) + a_y
        w = torch.exp(d_w / 5.0) * a_w
        h = torch.exp(d_h / 5.0) * a_h

        x1 = cx - (w / 2.0)
        x2 = cx + (w / 2.0)
        y1 = cy - (h / 2.0)
        y2 = cy + (h / 2.0)
        
        cls = self.activation(cls)

        return torch.concat([cls, x1, y1, x2, y2], dim=1).contiguous()
    
    
    