import random
import cv2
import numpy as np
import math
import copy


Background_color = 0

"""
img : 0~255 RGB Uint8, numpy.array
"""


class Resize(object):
    # box2d : no change
    # box3d : no change
    # polygon : no change
    # keypoint : no change
    def __init__(self, size):
        self.imh = size[0]
        self.imw = size[1]
        
    def __call__(self, img, label):
        img = cv2.resize(img, (self.imw, self.imh))
        
        return img, label


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
        

class RandomVFlip(object):
    # box2d : change
    # box3d : change (TODO)
    # polygon : change
    # keypoint : change (TODO)
    
    def __init__(self, p):
        self.p = p
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        img = cv2.flip(img, 1)
        
        for obj in label['object']:
            if 'box2d' in obj:
                # cx flip
                obj['box2d']['cx'] = 1 - obj['box2d']['cx']
            if 'polygon' in obj:
                for poly in obj['polygon']:
                    for point in poly:
                        point[0] = 1 - point[0]
            
        return img, label

class RandomHFlip(object):
    # box2d : change
    # box3d : change (TODO)
    # polygon : change
    # keypoint : change (TODO)
    
    def __init__(self, p):
        self.p = p
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        img = cv2.flip(img, 0)
        
        for obj in label['object']:
            if 'box2d' in obj:
                # cy flip
                obj['box2d']['cy'] = 1 - obj['box2d']['cy']
            if 'polygon' in obj:
                for poly in obj['polygon']:
                    for point in poly:
                        point[1] = 1 - point[1]
            
        return img, label


class augment_hsv(object):
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4, p=0.5):
        self.p = p
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        
        return img_hsv, label


class random_perspective(object):
    # box2d : change
    # box3d : change (TODO)
    # polygon : change
    # keypoint : change (TODO)
    
    def __init__(self, degree=10, translate=.1, scale=.1, shear=5, perspective=0.0, background=(0.5, 0.5, 0.5), p=0.5):
        self.degrees = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.background = np.array(background) * 255
        self.p = p

    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        height = img.shape[0]  # shape(h,w,c)
        width = img.shape[1]

        while_cnt = 0
        while while_cnt < 100:
            label_new = copy.deepcopy(label)
            while_cnt += 1
            # Center
            C = np.eye(3)
            C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
            C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

            # Perspective
            P = np.eye(3)
            P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
            P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

            # Rotation and Scale
            R = np.eye(3)
            a = random.uniform(-self.degrees, self.degrees)
            # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
            s = random.uniform(1 - self.scale, 1.1 + self.scale)
            # s = 2 ** random.uniform(-scale, scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

            # Translation
            T = np.eye(3)
            T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
            T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

            # Combined rotation matrix
            M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
            img_new = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=self.background)

            objects_new = []
            # Transform label coordinates
            for obj in label_new['object']:
                if 'box2d' in obj and 'polygon' not in obj:
                    cx = obj['box2d']['cx']
                    cy = obj['box2d']['cy']
                    w = obj['box2d']['w']
                    h = obj['box2d']['h']
                    x1 = (cx - (w / 2.0)) * width
                    x2 = (cx + (w / 2.0)) * width
                    y1 = (cy - (h / 2.0)) * height
                    y2 = (cy + (h / 2.0)) * height
                    
                    w_origin = (x2 - x1) / width
                    h_origin = (y2 - y1) / height
                    
                    xy = [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]
                    
                    xy = xy @ M.T  # transform
                    xy = xy[:, :2] / xy[:, 2:3]

                    # create new boxes
                    x = xy[:, 0]
                    y = xy[:, 1]
                    
                    x = np.clip(x, 0, width)
                    y = np.clip(y, 0, height)

                    x1 = x.min(0)
                    x2 = x.max(0)
                    y1 = y.min(0)
                    y2 = y.max(0)
                    
                    cx = (x1 + x2) / 2.0 / width
                    cy = (y1 + y2) / 2.0 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    if (w_origin * 0.5) > w or (h_origin * 0.5) > h:
                        continue
                    
                    obj_new = copy.deepcopy(obj)
                    
                    obj_new['box2d']['cx'] = cx
                    obj_new['box2d']['cy'] = cy
                    obj_new['box2d']['w'] = w
                    obj_new['box2d']['h'] = h
                    
                    objects_new.append(obj_new)
                    
                if 'polygon' in obj:
                    obj_new = copy.deepcopy(obj)
                    obj_new['polygon'] = []
                    obj_new['box2d'] = dict()
                    
                    mask = np.zeros((height, width), dtype=np.uint8)
                    for poly in obj['polygon']:
                        pts = np.array(poly, np.float32)
                        if pts.ndim == 3:
                            pts = pts.reshape(-1, 2)
                        pts[:, 0] = pts[:, 0] * width
                        pts[:, 1] = pts[:, 1] * height
                        
                        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(1))
                    
                    area_before = mask.sum()
                    mask_new = cv2.warpAffine(mask, M[:2], dsize=(width, height), borderValue=(0))
                    area_after = mask_new.sum()
                    
                    if area_after / area_before < 0.3:
                        continue
                    
                    contours, hierarchy = cv2.findContours(mask_new, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    
                    xmin = 1
                    xmax = 0
                    ymin = 1
                    ymax = 0
                    for i in range(len(contours)):
                        new_poly = contours[i].astype(np.float32).reshape(-1, 2)
                        new_poly[:, 0] = new_poly[:, 0] / width
                        new_poly[:, 1] = new_poly[:, 1] / height
                        
                        xmin = min(xmin, new_poly[:, 0].min())
                        xmax = max(xmax, new_poly[:, 0].max())
                        ymin = min(ymin, new_poly[:, 1].min())
                        ymax = max(ymax, new_poly[:, 1].max())
                        
                        new_poly = new_poly.tolist()
                        
                        obj_new['polygon'].append(new_poly)
                        
                    obj_new['box2d']['cx'] = (xmin + xmax) / 2.0
                    obj_new['box2d']['cy'] = (ymin + ymax) / 2.0
                    obj_new['box2d']['w'] = xmax - xmin
                    obj_new['box2d']['h'] = ymax - ymin
                    
                    if obj_new['box2d']['w'] < 0.01 or obj_new['box2d']['h'] < 0.01:
                        continue
                    
                    objects_new.append(obj_new)
                    
            
            if len(objects_new) == 0:
                continue
            
            label_new['object'] = objects_new

            return img_new, label_new
        return img, label

class mosaic(object):
    def __init__(self, canvas_range=1.0, p=0.5):
        self.canvas_range = canvas_range
        self.background = (128.0, 128.0, 128.0)
        self.p = p

        self.bank_img = []
        self.bank_label = []
        self.bank_boxes = []

    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        self.push_bank(img, label)

        if len(self.bank_img) < 4: # 4개 미만이면 리턴
            return img, label

        if random.random() > self.p:
            return img, label
        
        height = img.shape[0]  # shape(h,w,c)
        width = img.shape[1]

        # background
        img_new = np.zeros((height*3, width*3, 3), dtype=np.uint8) + np.array(self.background, dtype=np.uint8)

        # 4개 이미지 랜덤으로 뽑기
        img_list = []
        label_list = []
        for _ in range(4):
            img, label = self.pop_bank()
            img_list.append(img)
            label_list.append(label)

        # 4개 이미지 랜덤으로 위치 지정
        idx_list = [0, 1, 2, 3]
        random.shuffle(idx_list)

        # canvas에 이미지 붙이기
        random_range = self.canvas_range
        offset_width = 0.5 * width
        offset_width = offset_width + ((2*random.random() - 1.0) * offset_width * random_range)
        offset_height = 0.5 * height
        offset_height = offset_height + ((2*random.random() - 1.0) * offset_height * random_range)

        object_new = dict()
        object_new['object'] = []
        for i, idx in enumerate(idx_list):
            img = img_list[i]
            label = label_list[i]

            # order 
            # 0 1
            # 2 3

            if idx == 0:
                xmin = offset_width
                ymin = offset_height
            elif idx == 1:
                xmin = width + offset_width
                ymin = offset_height
            elif idx == 2:
                xmin = offset_width
                ymin = height + offset_height
            elif idx == 3:
                xmin = width + offset_width
                ymin = height + offset_height

            img_new[int(ymin):int(ymin+height), int(xmin):int(xmin+width), :] = img

            
            # Transform label coordinates
            for obj in label['object']:
                cx = obj['box2d']['cx']
                cy = obj['box2d']['cy']
                w = obj['box2d']['w']
                h = obj['box2d']['h']
                
                # cx, cy, w, h -> x1, y1, x2, y2
                # coordiante of original image
                x1_origin = (cx - (w / 2.0)) * width
                x2_origin = (cx + (w / 2.0)) * width
                y1_origin = (cy - (h / 2.0)) * height
                y2_origin = (cy + (h / 2.0)) * height

                # x1, y1, x2, y2 -> x1, y1, x2, y2
                # original image -> canvas image(2x)
                x1_canvas = x1_origin + xmin - (width * 0.5)
                x2_canvas = x2_origin + xmin - (width * 0.5)
                y1_canvas = y1_origin + ymin - (height * 0.5)
                y2_canvas = y2_origin + ymin - (height * 0.5)

                cx = (x1_canvas + x2_canvas) / 2.0 / (width * 2)
                cy = (y1_canvas + y2_canvas) / 2.0 / (height * 2)

                # filter out boxes
                if cx < 0 or cy < 0:
                    continue
                if cx > 1 or cy > 1:
                    continue
                
                # adjust boxes
                x1_canvas = np.clip(x1_canvas, 0, width * 2)
                x2_canvas = np.clip(x2_canvas, 0, width * 2)
                y1_canvas = np.clip(y1_canvas, 0, height * 2)
                y2_canvas = np.clip(y2_canvas, 0, height * 2)

                # x1, y1, x2, y2 -> cx, cy, w, h
                cx = (x1_canvas + x2_canvas) / 2.0 / (width * 2)
                cy = (y1_canvas + y2_canvas) / 2.0 / (height * 2)
                w = (x2_canvas - x1_canvas) / (width * 2)
                h = (y2_canvas - y1_canvas) / (height * 2)

                append_obj = copy.deepcopy(obj)
                append_obj['box2d']['cx'] = cx
                append_obj['box2d']['cy'] = cy
                append_obj['box2d']['w'] = w
                append_obj['box2d']['h'] = h
                
                object_new['object'].append(append_obj)
                

        img_xmin = 0.5 * width
        img_ymin = 0.5 * height
        img_xmax = 0.5 * width + (width * 2)
        img_ymax = 0.5 * height + (height * 2)

        img = img_new[int(img_ymin):int(img_ymax), int(img_xmin):int(img_xmax)]

        return img, object_new

    #@numba.jit(nopython=True, cache=True)
    def push_bank(self, img, label):
        if len(self.bank_img) > 40:
            return
        
        self.bank_img.append(img)
        self.bank_label.append(label)

    #@numba.jit(nopython=True, cache=True)
    def pop_bank(self):
        if len(self.bank_img) == 0:
            return None
        
        idx = random.randint(0, len(self.bank_img) - 1)
        
        return self.bank_img.pop(idx), self.bank_label.pop(idx)


class Oneof(object):
    def __init__(self, auglist, p=0.5):
        self.p = p
        self.auglist = auglist
        self.na = len(auglist) - 1
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label
        
        idx = random.randint(0, self.na)
        
        img, label = self.auglist[idx](img, label)
            
        return img, label


if __name__ == "__main__":
    import json
    
    class augmentator(object):
        def __init__(self, cfg):
            input_size = cfg['network']['input_size']

            hsv_cfg = cfg['augmentation']['hsv']
            mosaic_cfg = cfg['augmentation']['mosaic']
            flip_cfg = cfg['augmentation']['flip']
            random_zoomout_cfg = cfg['augmentation']['zoomout']
            random_crop_cfg = cfg['augmentation']['crop']
            perspective_cfg = cfg['augmentation']['perspective']

            mean = cfg['network']['mean']
            std = cfg['network']['std']

            self.transform_train = [
                RandomVFlip(p=flip_cfg['prob']),
                
                # photometric
                augment_hsv(hgain=hsv_cfg['hgain'], 
                            sgain=hsv_cfg['sgain'], 
                            vgain=hsv_cfg['vgain'], 
                            p=hsv_cfg['prob']),
                
                # geometric
                RandomCrop(min_overlap=random_crop_cfg['min_overlap'], 
                        p=random_crop_cfg['prob']),
                RandomZoomOut(max_scale=random_zoomout_cfg['max_ratio'],
                            mean=mean,
                            p=random_zoomout_cfg['prob']),
                
                # mosaic(canvas_range=mosaic_cfg['canvas_range'], 
                #     p=mosaic_cfg['prob']),
                random_perspective(degree=perspective_cfg['degree'], 
                                translate=perspective_cfg['translate'], 
                                scale=perspective_cfg['scale'], 
                                shear=perspective_cfg['shear'], 
                                perspective=perspective_cfg['perspective'], 
                                background=mean,
                                p=perspective_cfg['prob']),
                
                # resize
                Resize(input_size),
                ]

            self.transform_valid = [
                Resize(input_size),
                RandomVFlip(p=0.5),
                ]
                
        def __call__(self, img, labels):

            for tform in self.transform_train:
                img, labels = tform(img, labels)

            
            return img, labels

    #read json config
    with open('model/config/ssd/mobilenetv2_ssd_COCO2017.json') as config_file:
        config = json.load(config_file)
    
    aug = augmentator(config)
    
    sample_list = ['000000000785', '000000001000', '000000051976', '000000019432']
    # sample_list = ['000000019432']
    
    cnt = 0
    while True:
        sample = random.choice(sample_list)
        
        img = cv2.imread(f"sample/{sample}.jpg")
        with open(f"sample/{sample}.json") as f:
            labels = json.load(f)


        cnt += 1

        img, labels = aug(img, labels)
        img = cv2.resize(img, (300, 300))
        imh, imw, _ = img.shape
        
        label_max = 3
        print(len(labels['object']))
        for obj in labels['object']:
            if 'box2d' in obj:
                cx = obj['box2d']['cx']
                cy = obj['box2d']['cy']
                w = obj['box2d']['w']
                h = obj['box2d']['h']
                
                assert w > 0, 'w <= 0'
                assert h > 0, 'h <= 0'
                
                x1 = int(imw * (cx - (w / 2.0)))
                x2 = int(imw * (cx + (w / 2.0)))
                y1 = int(imh * (cy - (h / 2.0)))
                y2 = int(imh * (cy + (h / 2.0)))
            
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            if 'polygon' in obj:
                for poly in obj['polygon']:
                    pts = np.array(poly, np.float32)
                    pts = pts.reshape((-1, 2))
                    pts[:, 0] *= imw
                    pts[:, 1] *= imh
                    pts = pts.astype(np.int32)
                    cv2.polylines(img, [pts], True, (255, 0, 0), 1)
        
        # print(img[0, 0, :], np.max(np.max(np.max(img))), type(img), img.dtype)
        
        cv2.imshow('aug', img)
        cv2.waitKey(0)
        
    