import random
import cv2
import numpy as np
import math
import copy

class RandomZoomOut(object):
    # box2d : change
    # box3d : change (TODO)
    # polygon : change
    # keypoint : change (TODO)
    def __init__(self, max_scale, mean, p):
        self.p = p
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        while_cnt = 0
        while while_cnt < 100:
            while_cnt += 1
            
            label_new = copy.deepcopy(label)
            
            zoom_out_scale = None
            
            min_wh = 1
            for obj in label_new['object']:
                if 'box2d' in obj:
                    w = obj['box2d']['w']
                    h = obj['box2d']['h']
                    
                    min_wh = min(min_wh, w, h)
                    
            if min_wh < 0.05:
                zoom_out_scale = 1.0
            else:
                zoom_out_scale = min(min_wh / 0.05, self.max_scale)
            
            if zoom_out_scale is None:
                zoom_out_scale = self.max_scale
            
            original_h, original_w, _ = img.shape
            scale = random.uniform(1, zoom_out_scale)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)

            background = np.zeros((new_h, new_w, 3), dtype=np.uint8) + np.multiply(self.mean, 255).astype(np.uint8)

            left = int(random.uniform(0, new_w - original_w))
            right = left + original_w
            top = int(random.uniform(0, new_h - original_h))
            bottom = top + original_h
            
            background[top:bottom, left:right, :] = img

            
            for obj in label_new['object']:
                if 'box2d' in obj:
                    cx = obj['box2d']['cx']
                    cy = obj['box2d']['cy']
                    w = obj['box2d']['w']
                    h = obj['box2d']['h']
                    
                    # cx, cy, w, h -> x1, y1, x2, y2
                    x1 = (cx - w / 2.0) * original_w
                    y1 = (cy - h / 2.0) * original_h
                    x2 = (cx + w / 2.0) * original_w
                    y2 = (cy + h / 2.0) * original_h

                    # transform 
                    x1 = x1 + left
                    y1 = y1 + top
                    x2 = x2 + left
                    y2 = y2 + top

                    # x1, y1, x2, y2 -> cx, cy, w, h
                    obj['box2d']['cx'] = (x1 + x2) / 2.0 / new_w
                    obj['box2d']['cy'] = (y1 + y2) / 2.0 / new_h
                    obj['box2d']['w'] = (x2 - x1) / new_w
                    obj['box2d']['h'] = (y2 - y1) / new_h
                    
                if 'polygon' in obj:
                    for poly in obj['polygon']:
                        aspect_width = original_w / new_w
                        aspect_height = original_h / new_h
                        
                        offset_left = left / new_w
                        offset_top = top / new_h
                        
                        for i in range(len(poly)):
                            poly[i][0] = (poly[i][0] * aspect_width + offset_left)
                            poly[i][1] = (poly[i][1] * aspect_height + offset_top)
            

            img = cv2.resize(background, (original_w, original_h))

            return img, label_new
        
        return img, label
