import numpy as np

class preprocessor(object):
    def __init__(self, config, anchor):
        self.anchor = anchor
        self.nc = len(config['CLASS'])
    
    def __call__(self, label, box):
        gt = self.get_delta(label, box)
        return gt

    def set_nc(self, nc):
        self.nc = nc
        
    def get_delta(self, label, box):
        ngt = len(label)
        
        gtbox = np.array(box)
        conf_table = np.eye(self.nc)
        conf = []
        for i in range(ngt):
            conf.append(conf_table[label[i]])
        conf = np.array(conf)
        
        ious = self.jaccard_iou(self.anchor, gtbox) # (n_anchor, ngt)
        
        #assure at least 1 gt box
        assure_iou = np.argmax(ious, axis=0) # from gt to anchor
        
        maxidx = np.argmax(ious, axis=1) # from anchor to gt
        maxidx[assure_iou] = np.arange(ngt)

        maxiou = np.max(ious, axis=1)
        maxiou[assure_iou] = 1.0

        pos = maxiou > 0.5
        pos = np.expand_dims(pos, axis=1)
        # class
        conf = conf[maxidx] * pos
        conf = np.concatenate([1-pos, conf], axis=1) # backgorund append
        
        # box regression
        target = gtbox[maxidx]
        
        delta_cx = np.expand_dims((target[:, 0] - self.anchor[:, 0]) / self.anchor[:, 2], axis=1) * 10.0
        delta_cy = np.expand_dims((target[:, 1] - self.anchor[:, 1]) / self.anchor[:, 3], axis=1) * 10.0
        delta_w = np.expand_dims(np.log(target[:, 2] / self.anchor[:, 2]), axis=1) * 5.0
        delta_h = np.expand_dims(np.log(target[:, 3] / self.anchor[:, 3]), axis=1) * 5.0
        
        delta = np.concatenate([delta_cx, delta_cy, delta_w, delta_h], axis=1)
        delta = delta * pos
        
        output = np.concatenate([conf, delta], axis=1)
        return output

    def jaccard_iou(self, box1, box2):
        # box1 : (cx, cy, w, h) anchor (nbox, 4)
        # box2 : (cx, cy, w, h) gtbox (ngt, 4)
        # return : (nbox, ngt)
        nbox = np.size(box1, 0)
        ngt = np.size(box2, 0)

        box1 = np.expand_dims(box1, axis=1)
        box2 = np.expand_dims(box2, axis=0)
        
        box1 = np.tile(box1, (1, ngt, 1)) # anchor (nbox, ngt, 4)
        box2 = np.tile(box2, (nbox, 1, 1)) # gtbox (nbox, ngt, 4)
        
        x1_box1 = box1[:, :, 0] - (box1[:, :, 2] / 2.0)
        x2_box1 = box1[:, :, 0] + (box1[:, :, 2] / 2.0)
        y1_box1 = box1[:, :, 1] - (box1[:, :, 3] / 2.0)
        y2_box1 = box1[:, :, 1] + (box1[:, :, 3] / 2.0)
        
        x1_box2 = box2[:, :, 0] - (box2[:, :, 2] / 2.0)
        x2_box2 = box2[:, :, 0] + (box2[:, :, 2] / 2.0)
        y1_box2 = box2[:, :, 1] - (box2[:, :, 3] / 2.0)
        y2_box2 = box2[:, :, 1] + (box2[:, :, 3] / 2.0)

        xmin = np.maximum(x1_box1, x1_box2)
        xmax = np.minimum(x2_box1, x2_box2)
        ymin = np.maximum(y1_box1, y1_box2)
        ymax = np.minimum(y2_box1, y2_box2)

        area1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1) # anchor (nbox, ngt)
        area2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2) # gtbox (nbox, ngt)
        
        common = np.maximum((xmax - xmin), 0.0) * np.maximum((ymax - ymin), 0.0)

        iou = common / (area1 + area2 - common)

        return iou