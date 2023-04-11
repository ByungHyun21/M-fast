import random
import cv2
import numpy as np
import math

import torch
from torchvision import transforms

Background_color = 0

"""
img : 0~255 RGB Uint8, numpy.array
label : int list. ex) [2, 1, 2, 2, 0]
boxes : float list, 0~1, cx, cy, w, h
        ex) [[0.1, 0.2, 0.2, 0.3], [0.3, 0.89, 0.3, 0.4]]
"""

class Resize(object):
    def __init__(self, size):
        self.imh = size[0]
        self.imw = size[1]
        
    def __call__(self, img, label, boxes):
        img = cv2.resize(img, (self.imw, self.imh))
        
        return img, label, boxes
    
class RandomZoomOut(object):
    def __init__(self, max_scale, mean, p):
        self.p = p
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        original_h, original_w, _ = img.shape
        scale = random.uniform(1, self.max_scale)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        background = np.zeros((new_h, new_w, 3), dtype=np.uint8) + np.multiply(self.mean, 255).astype(np.uint8)

        left = int(random.uniform(0, new_w - original_w))
        right = left + original_w
        top = int(random.uniform(0, new_h - original_h))
        bottom = top + original_h
        
        background[top:bottom, left:right, :] = img

        for i in range(len(boxes)):
            # cx, cy, w, h -> x1, y1, x2, y2
            x1 = (boxes[i][0] - boxes[i][2] / 2.0) * original_w
            y1 = (boxes[i][1] - boxes[i][3] / 2.0) * original_h
            x2 = (boxes[i][0] + boxes[i][2] / 2.0) * original_w
            y2 = (boxes[i][1] + boxes[i][3] / 2.0) * original_h

            # transform 
            x1 = x1 + left
            y1 = y1 + top
            x2 = x2 + left
            y2 = y2 + top

            # x1, y1, x2, y2 -> cx, cy, w, h
            boxes[i][0] = (x1 + x2) / 2.0 / new_w
            boxes[i][1] = (y1 + y2) / 2.0 / new_h
            boxes[i][2] = (x2 - x1) / new_w
            boxes[i][3] = (y2 - y1) / new_h

        img = cv2.resize(background, (original_w, original_h))

        return img, label, boxes
    
class RandomCrop(object):
    # 기능상 Random zoom in과 동일
    def __init__(self, min_overlap, p):
        self.min_overlap = min_overlap
        self.p = p

    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        original_h, original_w, _ = img.shape

        max_trials = 50
        for _ in range(max_trials):
            crop_w = random.uniform(0.3, 1) * original_w
            crop_h = random.uniform(0.3, 1) * original_h

            aspect_ratio = crop_h / crop_w
            if aspect_ratio < 0.5 or aspect_ratio > 2:
                continue

            left = random.uniform(0, original_w - crop_w)
            top = random.uniform(0, original_h - crop_h)

            new_label = []
            new_boxes = []
            for idx, box in enumerate(boxes):
                # cx, cy, w, h -> x1, y1, x2, y2
                x1_origin = (box[0] - box[2] / 2.0) * original_w
                y1_origin = (box[1] - box[3] / 2.0) * original_h
                x2_origin = (box[0] + box[2] / 2.0) * original_w
                y2_origin = (box[1] + box[3] / 2.0) * original_h

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

                    new_label.append(label[idx])
                    new_boxes.append([new_cx, new_cy, new_w, new_h])
            
            if len(new_boxes) > 0:
                new_img = img[int(top):int(top + crop_h), int(left):int(left + crop_w), :]
                new_img = cv2.resize(new_img, (original_w, original_h))
                return new_img, new_label, new_boxes

        return img, label, boxes
        
        
class RandomVFlip(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        img = cv2.flip(img, 1)
        
        for i in range(len(boxes)):
            # cx flip
            boxes[i][0] = 1 - boxes[i][0]
            
        return img, label, boxes

class RandomHFlip(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        img = cv2.flip(img, 0)
        
        for i in range(len(boxes)):
            # cy flip
            boxes[i][1] = 1 - boxes[i][1]
            
        return img, label, boxes

class augment_hsv(object):
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4, p=0.5):
        self.p = p
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        
        return img_hsv, label, boxes

class random_perspective(object):
    def __init__(self, degree=10, translate=.1, scale=.1, shear=5, perspective=0.0, p=0.5):

        self.degrees = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.background = (128, 128, 128)
        self.p = p

    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        height = img.shape[0]  # shape(h,w,c)
        width = img.shape[1]

        while True:
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

            label_new = []
            boxes_new = []
            # Transform label coordinates
            for idx, box in enumerate(boxes):
                x1 = (box[0] - (box[2] / 2.0)) * width
                x2 = (box[0] + (box[2] / 2.0)) * width
                y1 = (box[1] - (box[3] / 2.0)) * height
                y2 = (box[1] + (box[3] / 2.0)) * height
                
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
                
                label_new.append(label[idx])
                boxes_new.append([cx, cy, w, h])
            
            if len(label_new) == 0 or len(boxes_new) == 0:
                continue

            return img_new, label_new, boxes_new

class mosaic(object):
    def __init__(self, canvas_range=1.0, p=0.5):
        self.canvas_range = canvas_range
        self.background = (128.0, 128.0, 128.0)
        self.p = p

        self.bank_img = []
        self.bank_label = []
        self.bank_boxes = []

    def __call__(self, img, label, boxes):
        self.push_bank(img, label, boxes)

        if len(self.bank_img) < 4: # 4개 미만이면 리턴
            return img, label, boxes

        if random.random() > self.p:
            return img, label, boxes
        
        height = img.shape[0]  # shape(h,w,c)
        width = img.shape[1]

        # background
        img_new = np.zeros((height*3, width*3, 3), dtype=np.uint8) + np.array(self.background, dtype=np.uint8)

        # 4개 이미지 랜덤으로 뽑기
        img_list = []
        label_list = []
        boxes_list = []
        for _ in range(4):
            img, label, boxes = self.pop_bank()
            img_list.append(img)
            label_list.append(label)
            boxes_list.append(boxes)

        # 4개 이미지 랜덤으로 위치 지정
        idx_list = [0, 1, 2, 3]
        random.shuffle(idx_list)

        # canvas에 이미지 붙이기
        random_range = self.canvas_range
        offset_width = 0.5 * width
        offset_width = offset_width + ((2*random.random() - 1.0) * offset_width * random_range)
        offset_height = 0.5 * height
        offset_height = offset_height + ((2*random.random() - 1.0) * offset_height * random_range)

        label_new = []
        boxes_new = []
        for i, idx in enumerate(idx_list):
            img = img_list[i]
            label = label_list[i]
            boxes = boxes_list[i]

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
            for idx, box in enumerate(boxes):
                # cx, cy, w, h -> x1, y1, x2, y2
                # coordiante of original image
                x1_origin = (box[0] - (box[2] / 2.0)) * width
                x2_origin = (box[0] + (box[2] / 2.0)) * width
                y1_origin = (box[1] - (box[3] / 2.0)) * height
                y2_origin = (box[1] + (box[3] / 2.0)) * height

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

                

                label_new.append(label[idx])
                boxes_new.append([cx, cy, w, h])

        img_xmin = 0.5 * width
        img_ymin = 0.5 * height
        img_xmax = 0.5 * width + (width * 2)
        img_ymax = 0.5 * height + (height * 2)

        img = img_new[int(img_ymin):int(img_ymax), int(img_xmin):int(img_xmax)]

        return img, label_new, boxes_new

    def push_bank(self, img, label, boxes):
        if len(self.bank_img) > 40:
            return
        
        self.bank_img.append(img)
        self.bank_label.append(label)
        self.bank_boxes.append(boxes)

    def pop_bank(self):
        if len(self.bank_img) == 0:
            return None
        
        idx = random.randint(0, len(self.bank_img) - 1)
        
        return self.bank_img.pop(idx), self.bank_label.pop(idx), self.bank_boxes.pop(idx)


class Oneof(object):
    def __init__(self, auglist, p=0.5):
        self.p = p
        self.auglist = auglist
        self.na = len(auglist) - 1
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        idx = random.randint(0, self.na)
        
        img, label, boxes = self.auglist[idx](img, label, boxes)
            
        return img, label, boxes


if __name__ == "__main__":
    class augmentator(object):
        def __init__(self, config):
            input_size = config['INPUT_SIZE']

            hsv_prob = config['HSV_PROB']
            hsv_hgain = config['HSV_HGAIN']
            hsv_sgain = config['HSV_SGAIN']
            hsv_vgain = config['HSV_VGAIN']

            mosaic_prob = config['MOSAIC_PROB']
            mosaic_canvas_range = config['MOSAIC_CANVAS_RANGE']

            perspective_prob = config['PERSPECTIVE_PROB']
            perspective_degree = config['PERSPECTIVE_DEGREE']
            perspective_translate = config['PERSPECTIVE_TRANSLATE']
            perspective_scale = config['PERSPECTIVE_SCALE']
            perspective_shear = config['PERSPECTIVE_SHEAR']
            perspective_perspective = config['PERSPECTIVE_PERSPECTIVE']

            random_crop_prob = config['RANDOM_CROP_PROB']
            random_crop_min_overlap = config['RANDOM_CROP_MIN_OVERLAP']
            
            random_zoomout_prob = config['RANDOM_ZOOMOUT_PROB']
            random_zoomout_max_scale = config['RANDOM_ZOOMOUT_MAX_SCALE']
            

            mean = config['MEAN']
            std = config['STD']

            self.transform = [
                # photometric
                augment_hsv(hgain=hsv_hgain, sgain=hsv_sgain, vgain=hsv_vgain, p=hsv_prob),

                # resize
                Resize(input_size),

                # geometric
                RandomZoomOut(max_scale=random_zoomout_max_scale, mean=mean, p=random_zoomout_prob),
                RandomCrop(min_overlap=random_crop_min_overlap, p=random_crop_prob),
                mosaic(canvas_range=mosaic_canvas_range, p=mosaic_prob),
                random_perspective(degree=perspective_degree, translate=perspective_translate, scale=perspective_scale, shear=perspective_shear, perspective=perspective_perspective, p=perspective_prob),
                
                RandomVFlip(p=0.5),
                # resize
                Resize(input_size),
                
                ]

                
        def __call__(self, img, labels, boxes):
            for tform in self.transform:
                img, labels, boxes = tform(img, labels, boxes)

            return img, labels, boxes

    config = {'INPUT_SIZE': [600, 600],
              'HSV_PROB': 1.0,
              'HSV_HGAIN': 0.015,
              'HSV_SGAIN': 0.7,
              'HSV_VGAIN': 0.4,
              'MOSAIC_PROB': 1.0,
              'MOSAIC_CANVAS_RANGE': 1.0,
              'PERSPECTIVE_PROB': 0.0,
              'PERSPECTIVE_DEGREE': 10,
              'PERSPECTIVE_TRANSLATE': 0.1,
              'PERSPECTIVE_SCALE': 0.1,
              'PERSPECTIVE_SHEAR': 2.0,
              'PERSPECTIVE_PERSPECTIVE': 0.0,
              
              'RANDOM_ZOOMOUT_PROB': 0.0,
              'RANDOM_ZOOMOUT_MAX_SCALE': 4.0,

              'RANDOM_CROP_PROB': 1.0,
              'RANDOM_CROP_MIN_OVERLAP': 0.3,

              'MEAN': [0.485, 0.456, 0.406],
              'STD': [0.229, 0.224, 0.225]}
    
    aug = augmentator(config)
    
    cnt = 0
    while True:
        if cnt % 2 == 0:
            img = cv2.imread('sample/augmentation_test_sample.jpg')
            labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3]
            boxes = [
                    [0.587203125, 0.29015625, 0.08225, 0.0856875],
                    [0.127765625, 0.6070520833333333, 0.09259375, 0.44277083333333334],
                    [0.3613828125, 0.53690625, 0.149640625, 0.5682291666666667],
                    [0.7106171875, 0.34918750000000004, 0.13764062500000002, 0.2524166666666667],
                    [0.5806171874999999, 0.6483125, 0.129765625, 0.6539166666666667],
                    [0.6641718750000001, 0.6656979166666666, 0.1381875, 0.6651041666666667],
                    [0.730703125, 0.71721875, 0.17953125, 0.5655625000000001],
                    [0.8942734375000001, 0.6999479166666667, 0.21145312500000002, 0.6001041666666667],
                    [0.4047578124999999, 0.6232916666666667, 0.155671875, 0.5189166666666667],
                    [0.48404687499999993, 0.52875, 0.1389375, 0.6580833333333334],
                    [0.6632734375, 0.2982916666666667, 0.058015625, 0.09483333333333334],
                    [0.2449609375, 0.5548645833333333, 0.130046875, 0.47585416666666663],
                    [0.3625625, 0.58984375, 0.10978125000000001, 0.24489583333333334],
                    [0.07372656250000001, 0.5989062500000001, 0.083265625, 0.2531875],
                    [0.1106015625, 0.7226041666666667, 0.074109375, 0.18191666666666664],
                    [0.11028906249999999, 0.5195416666666667, 0.08707812499999999, 0.10620833333333332],
                    [0.37609375, 0.40187500000000004, 0.09753125, 0.12091666666666666]
            ]
            cnt = 0
        else:
            img = cv2.imread('sample/augmentation_test_sample_2.jpg')
            labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4]
            boxes = [
                    [0.1434, 0.6993693693693694, 0.22003999999999999, 0.5467267267267267],
                    [0.15138, 0.33708708708708707, 0.10636, 0.18852852852852853],
                    [0.08592, 0.3695945945945946, 0.10164, 0.1569069069069069],
                    [0.11599, 0.021351351351351352, 0.07334, 0.042702702702702704],
                    [0.47817, 0.31012012012012014, 0.0943, 0.16180180180180181],
                    [0.8623999999999999, 0.26076576576576577, 0.17628, 0.16051051051051052],
                    [0.39127, 0.30890390390390393, 0.10526, 0.17084084084084084],
                    [0.24067, 0.3182432432432432, 0.13306, 0.2074174174174174],
                    [0.32852, 0.7648348348348348, 0.30760000000000004, 0.42150150150150156],
                    [0.61075, 0.5626576576576577, 0.19413999999999998, 0.6995495495495495],
                    [0.29982, 0.3328978978978979, 0.01064, 0.047177177177177176],
                    [0.45612, 0.7415615615615615, 0.0576, 0.09933933933933933],
                    [0.6659400000000001, 0.18978978978978978, 0.12448000000000001, 0.2094294294294294],
                    [0.76038, 0.32566066066066063, 0.15708000000000003, 0.06363363363363364],
                    [0.33015, 0.3766516516516517, 0.10729999999999999, 0.056726726726726726]
            ]

        cnt += 1

        img, labels, boxes = aug(img, labels, boxes)
        img = cv2.resize(img, (600, 600))
        h, w, _ = img.shape
        
        label_max = 3
        for label, box in zip(labels, boxes):
            x1 = int(w * (box[0] - (box[2] / 2.0)))
            x2 = int(w * (box[0] + (box[2] / 2.0)))
            y1 = int(h * (box[1] - (box[3] / 2.0)))
            y2 = int(h * (box[1] + (box[3] / 2.0)))
        
            color = (0, int(255//label_max * label), int(255 - 255//label_max * label))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        print(img[0, 0, :], np.max(np.max(np.max(img))), type(img), img.dtype)
        
        cv2.imshow('aug', img)
        cv2.waitKey(0)
        
    