import random
import cv2
import numpy as np
import math
import copy

class RandomCrop(object):
    # box2d : change
    # box3d : change (TODO)
    # polygon : change
    # keypoint : change (TODO)
    
    # 기능상 Random zoom in과 동일
    def __init__(self, min_overlap, p):
        self.min_overlap = min_overlap
        self.p = p

    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        original_h, original_w, _ = img.shape

        while_cnt = 0
        while while_cnt < 100:
            while_cnt += 1
            crop_w = random.uniform(0.3, 1) * original_w
            crop_h = random.uniform(0.3, 1) * original_h

            aspect_ratio = crop_h / crop_w
            if aspect_ratio < 0.5 or aspect_ratio > 2:
                continue

            left = random.uniform(0, original_w - crop_w)
            top = random.uniform(0, original_h - crop_h)

            new_label = copy.deepcopy(label)
            new_label['object'] = []
            for obj in label['object']:
                if 'box2d' in obj and 'polygon' not in obj:
                    # cx, cy, w, h -> x1, y1, x2, y2
                    cx = obj['box2d']['cx']
                    cy = obj['box2d']['cy']
                    w = obj['box2d']['w']
                    h = obj['box2d']['h']
                    
                    x1_origin = (cx - w / 2.0) * original_w
                    y1_origin = (cy - h / 2.0) * original_h
                    x2_origin = (cx + w / 2.0) * original_w
                    y2_origin = (cy + h / 2.0) * original_h

                    x1 = max(x1_origin, left)
                    y1 = max(y1_origin, top)
                    x2 = min(x2_origin, left + crop_w)
                    y2 = min(y2_origin, top + crop_h)

                    if x2 < x1 or y2 < y1:
                        continue
                    
                    overlap_area = (x2 - x1) * (y2 - y1)
                    box_area = (x2_origin - x1_origin) * (y2_origin - y1_origin)
                    iou = overlap_area / box_area

                    if iou >= self.min_overlap:
                        # x1, y1, x2, y2 -> cx, cy, w, h
                        new_x1 = (x1 - left)
                        new_y1 = (y1 - top)
                        new_x2 = (x2 - left)
                        new_y2 = (y2 - top)

                        new_cx = (new_x1 + new_x2) / 2.0 / crop_w
                        new_cy = (new_y1 + new_y2) / 2.0 / crop_h
                        new_w = (new_x2 - new_x1) / crop_w
                        new_h = (new_y2 - new_y1) / crop_h

                        obj['box2d']['cx'] = new_cx
                        obj['box2d']['cy'] = new_cy
                        obj['box2d']['w'] = new_w
                        obj['box2d']['h'] = new_h
                        
                        if new_w < 0.01 or new_h < 0.01:
                            continue
                        
                        new_label['object'].append(copy.deepcopy(obj))  
                
                if 'polygon' in obj:                    
                    new_obj = copy.deepcopy(obj)
                    new_obj['box2d'] = []
                    new_obj['polygon'] = []
                    
                    min_x = 1
                    max_x = 0
                    min_y = 1
                    max_y = 0
                    
                    poly_mask = np.zeros((original_h, original_w), dtype=np.uint8)
                    for idx, poly in enumerate(obj['polygon']):
                        poly = np.array(poly, dtype=np.float32)
                        poly[:, 0] = poly[:, 0] * original_w
                        poly[:, 1] = poly[:, 1] * original_h
                        poly = poly.astype(np.int32)
                        cv2.fillPoly(poly_mask, [poly], 1)
                    
                    left_side = int(left)
                    right_side = int(left + crop_w)
                    top_side = int(top)
                    bottom_side = int(top + crop_h)
                    
                    area_before = np.sum(poly_mask)
                    area_after = np.sum(poly_mask[top_side:bottom_side, left_side:right_side])
                    
                    if area_after / area_before < self.min_overlap:
                        continue
                    
                    poly_mask = poly_mask[top_side:bottom_side, left_side:right_side]
                    
                    # mask to polygon
                    contours, hierarchy = cv2.findContours(poly_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    
                    for i in range(len(contours)):
                        new_poly = contours[i].astype(np.float32)
                        if new_poly.ndim == 3:
                            new_poly = new_poly.squeeze()
                        if new_poly.ndim < 2:
                            continue
                            
                        new_poly[:, 0] = new_poly[:, 0] / crop_w
                        new_poly[:, 1] = new_poly[:, 1] / crop_h
                        
                        new_obj['polygon'].append(new_poly.tolist())
                        
                        min_x = min(min_x, np.min(new_poly[:, 0]))
                        max_x = max(max_x, np.max(new_poly[:, 0]))
                        min_y = min(min_y, np.min(new_poly[:, 1]))
                        max_y = max(max_y, np.max(new_poly[:, 1]))
                        
                    new_obj['box2d'] = dict()
                    new_obj['box2d']['cx'] = (min_x + max_x) / 2.0
                    new_obj['box2d']['cy'] = (min_y + max_y) / 2.0
                    new_obj['box2d']['w'] = max_x - min_x
                    new_obj['box2d']['h'] = max_y - min_y
                    
                    if new_obj['box2d']['w'] < 0.01 or new_obj['box2d']['h'] < 0.01:
                        continue
                    
                    new_label['object'].append(new_obj)

            if len(new_label['object']) > 0:
                new_img = img[int(top):int(top + crop_h), int(left):int(left + crop_w), :]
                new_img = cv2.resize(new_img, (original_w, original_h))
                return new_img, new_label

        return img, label
        
