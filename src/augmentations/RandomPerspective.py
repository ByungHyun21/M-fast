import random
import cv2
import numpy as np
import math
import copy

class RandomPerspective(object):
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
