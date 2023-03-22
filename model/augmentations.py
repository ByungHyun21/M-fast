import random
import cv2
import numpy as np
import math

import torch
from torchvision import transforms

Background_color = 0

"""
img : 0~1 RGB float32, numpy.array
label : int list. ex) [2, 1, 2, 2, 0]
boxes : float list, 0~1, cx, cy, w, h
        ex) [[0.1, 0.2, 0.2, 0.3], [0.3, 0.89, 0.3, 0.4]]


         *** Input is 0~255 
         *** Normalize -> 0~255 change 0~1
         *** do Augmentation
         *** DeNormalize -> 0~1 change 0~255
"""

class Resize(object):
    def __init__(self, size):
        self.imh = size[0]
        self.imw = size[1]
        
    def __call__(self, img, label, boxes):
        img = cv2.resize(img, (self.imw, self.imh))
        
        return img, label, boxes

class Normalize(object):
    def __call__(self, img, label, boxes):
        img = (img / 255.0).astype(np.float32)
        
        return img, label, boxes

class DeNormalize(object):
    def __call__(self, img, label, boxes):
        img = (img * 255.0).astype(np.uint8)
        
        return img, label, boxes

class rgb2gray(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img, label, boxes

class RandomNoiseGaussian(object):
    def __init__(self, mean=0.01, variance=0.01, p=0.5):
        self.p = p
        self.mean = mean
        self.variance = variance
        self.sigma = self.variance ** 0.5
        
    def __call__(self, img, label, boxes):
        # img: 300x300x3 rgb (0~1)
        # label: [label, label, ... ], list of label (not one hot encoding)
        # box: [[cx, cy, w, h], [cx, cy, w, h], ... ], list of box (0~1)
        
        if random.random() > self.p:
            return img, label, boxes
        
        h, w, c = img.shape
        mean = self.mean * random.random()
        variance = self.variance * random.random()
        
        gaussnoise = np.random.normal(mean, variance, (h, w, c))
        gauss = gaussnoise.reshape(h, w, c)
        img = img + gauss
        
        img = np.clip(img, 0, 1)

        return img, label, boxes

class brighter(object):
    def __init__(self, brightness=0.5, p=0.5):
        self.p = p
        self.brightness = brightness
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        img = img * (1 + random.random() * self.brightness)
        img = np.clip(img, a_min=0.0, a_max=1.0)
        
        return img, label, boxes

class darker(object):
    def __init__(self, darkness=0.5, p=0.5):
        self.p = p
        self.darkness = darkness
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        img = img * (1 - random.random() * self.darkness)
        
        return img, label, boxes

class blurring(object):
    def __init__(self, patch_size=3, p=0.5):
        self.p = p
        self.patch_size = patch_size
        
    def __call__(self, img, label, boxes):
        # img: 300x300x3 rgb (0~1)
        # label: [label, label, ... ], list of label (not one hot encoding)
        # box: [[cx, cy, w, h], [cx, cy, w, h], ... ], list of box (0~1)
        if random.random() > self.p:
            return img, label, boxes
        
        kernel = np.ones((self.patch_size, self.patch_size), np.float32) / (self.patch_size ** 2)
        img = cv2.filter2D(img, -1, kernel)
        
        return img, label, boxes

class sharpening(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img, label, boxes):
        # img: 300x300x3 rgb (0~1)
        # label: [label, label, ... ], list of label (not one hot encoding)
        # box: [[cx, cy, w, h], [cx, cy, w, h], ... ], list of box (0~1)
        if random.random() > self.p:
            return img, label, boxes
        
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        
        return img, label, boxes

class RandomZoomIn(object):
    def __init__(self, p, zoom_range):
        """
        zoom_range 0.1 -> 10% zoom (270x270 -> 300x300)
        zoom_range 0.5 -> 50% zoom (150x150 -> 300x300)
        """
        self.p = p
        self.zoom_range = zoom_range
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        im_h, im_w, _ = img.shape
        
        xmin = 1
        xmax = 0
        ymin = 1
        ymax = 0
        for box in boxes:
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]
            
            x1 = cx - (w / 2.0)
            x2 = cx + (w / 2.0)
            y1 = cy - (h / 2.0)
            y2 = cy + (h / 2.0)
            
            xmin = min(xmin, x1)
            xmax = max(xmax, x2)
            ymin = min(ymin, y1)
            ymax = max(ymax, y2)
        
        xmin = max(min(xmin, 1), 0)
        xmax = max(min(xmax, 1), 0)
        ymin = max(min(ymin, 1), 0)
        ymax = max(min(ymax, 1), 0)

        crop_x1 = int((random.random() * self.zoom_range) * xmin * im_w)
        crop_x2 = int(((1 - random.random() * self.zoom_range) * (1 - xmax) + xmax) * im_w)
        crop_y1 = int((random.random() * self.zoom_range) * ymin * im_h)
        crop_y2 = int(((1 - random.random() * self.zoom_range) * (1 - ymax) + ymax) * im_h)

        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        size_w = im_w / crop_w
        size_h = im_h / crop_h
        size_x1 = crop_x1 / im_w
        size_y1 = crop_y1 / im_h
        
        newboxes = []
        for box in boxes:
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]
            
            new_w = w * size_w
            new_h = h * size_h
            new_cx = (cx - size_x1) * size_w
            new_cy = (cy - size_y1) * size_h
            
            newboxes.append([new_cx, new_cy, new_w, new_h])
            
        boxes = newboxes
        
        return img, label, boxes
    
class RandomZoomOut(object):
    def __init__(self, p, zoom_range):
        """
        zoom_range 0.1 -> 330x330 -> 300x300
        zoom_range 0.5 -> 450x450 -> 300x300
        """
        self.p = p
        self.zoom_range = zoom_range
        
    def __call__(self, img, label, boxes):
        # img: 300x300x3 rgb (0~1)
        # label: [label, label, ... ], list of label (not one hot encoding)
        # box: [[cx, cy, w, h], [cx, cy, w, h], ... ], list of box (0~1)
        if random.random() > self.p:
            return img, label, boxes
        
        im_h, im_w, im_c = img.shape
        img_new = np.ones((im_h, im_w, im_c), dtype=np.float32) * Background_color

        zoom_range_w = 1 - random.random() * self.zoom_range
        zoom_range_h = 1 - random.random() * self.zoom_range
        
        zoomout_w = min(int(im_w * zoom_range_w), im_w)
        zoomout_h = min(int(im_h * zoom_range_h), im_h)

        img_mask = cv2.resize(np.array(img, dtype=np.float32),
                              dsize=(zoomout_w, zoomout_h))
        
        im_h_mask, im_w_mask, _ = img_mask.shape
        ratio_h = im_h_mask / im_h
        ratio_w = im_w_mask / im_w
        
        sx = int(random.random() * (1 - ratio_w) * im_w)
        sy = int(random.random() * (1 - ratio_h) * im_h)

        # sx = int(0 * (1 - ratio_w) * im_w)
        # sy = int(0 * (1 - ratio_h) * im_h)

        img_new[sy:sy+im_h_mask, sx:sx+im_w_mask, :] = img_mask
        
        new_boxes = []
        for box in boxes:
            w = box[2] * ratio_w
            h = box[3] * ratio_h
            cx = box[0] * ratio_w + sx / im_w
            cy = box[1] * ratio_h + sy / im_h
            
            new_boxes.append([cx, cy, w, h])
        
        img = img_new
        boxes = new_boxes
        
        return img, label, boxes

class RandomRotation(object):
    def __init__(self, degree=5, p=0.5):
        self.p = p
        self.degree = degree
        
    def __call__(self, img, label, boxes):
        # img: 300x300x3 rgb (0~1)
        # label: [label, label, ... ], list of label (not one hot encoding)
        # box: [[cx, cy, w, h], [cx, cy, w, h], ... ], list of box (0~1)
        if random.random() > self.p:
            return img, label, boxes
        
        rot_deg = random.random() * 2 * self.degree - self.degree
        
        im_h, im_w, _ = img.shape
        im_cx = im_w // 2
        im_cy = im_h // 2
        
        rot_matrix = cv2.getRotationMatrix2D((im_cx, im_cy), -rot_deg, 1.0)
        img = cv2.warpAffine(img, rot_matrix, (im_w, im_h), 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=Background_color)
        for idx, box in enumerate(boxes):
            x1 = box[0] - box[2] / 2
            y1 = box[1] - box[3] / 2
            x2 = box[0] + box[2] / 2
            y2 = box[1] + box[3] / 2
            
            xmin = 1
            xmax = 0
            ymin = 1
            ymax = 0
            points = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]
            for p in points:
                angle = math.radians(rot_deg)
                x_ = 0.5 + math.cos(angle) * (p[0] - 0.5) - math.sin(angle) * (p[1] - 0.5)
                y_ = 0.5 + math.sin(angle) * (p[0] - 0.5) + math.cos(angle) * (p[1] - 0.5)
                xmin = min(xmin, x_)
                xmax = max(xmax, x_)
                ymin = min(ymin, y_)
                ymax = max(ymax, y_)
            
            xmin = max(min(xmin, 1), 0)
            xmax = max(min(xmax, 1), 0)
            ymin = max(min(ymin, 1), 0)
            ymax = max(min(ymax, 1), 0)
            
            w_ = max(min(xmax - xmin, 1), 1e-5)
            h_ = max(min(ymax - ymin, 1), 1e-5)
            cx_ = max(min((xmax + xmin) / 2, 1-1e-5), 1e-5)
            cy_ = max(min((ymax + ymin) / 2, 1-1e-5), 1e-5)
            
            boxes[idx] = [cx_, cy_, w_, h_]
        
        return img, label, boxes

class Random90Rotation(object):
    def __init__(self, d0=True, d90=True, d180=True, d270=True, p=0.5):
        self.p = p
        self.degree = [d0, d90, d180, d270]
        
    def __call__(self, img, label, boxes):
        idx = random.randint(0, 3)
        
        if self.degree[idx] and idx == 0:
            # 0 degree
            return img, label, boxes
        if self.degree[idx] and idx == 1:
            # 90 degree (clockwise)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            for i, box in enumerate(boxes):
                cx = 1 - box[1]
                cy = box[0]
                w = box[3]
                h = box[2]
                
                boxes[i] = [cx, cy, w, h]
            return img, label, boxes
        if self.degree[idx] and idx == 2:
            # 180 degree (clockwise)
            img = cv2.rotate(img, cv2.ROTATE_180)
            for i, box in enumerate(boxes):
                cx = 1 - box[0]
                cy = 1 - box[1]
                w = box[2]
                h = box[3]
                
                boxes[i] = [cx, cy, w, h]
            return img, label, boxes
        if self.degree[idx] and idx == 3:
            # 270 degree (clockwise)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for i, box in enumerate(boxes):
                cx = box[1]
                cy = 1 - box[0]
                w = box[3]
                h = box[2]
                
                boxes[i] = [cx, cy, w, h]
            
            return img, label, boxes

# TODO: 박스 내부에 임의의 지점을 지워버림 혹은 대체
class RandomErase(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        
        return img, label, boxes

# TODO: 박스 내부에 SxS 그리드를 만들고, 그리드 몇개를 지워버림
class RandomGridErase(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        
        return img, label, boxes

# TODO: 박스 내부에 SxS 그리드를 만들고, 그리드를 규칙적으로 제거함
class RandomGridMasking(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        
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
    def __init__(self, degrees=10, translate=.1, scale=.1, shear=5, perspective=0.0, p=0.5):

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.p = p

    def __call__(self, img, label, boxes):
        if random.random() > self.p:
            return img, label, boxes
        
        height = img.shape[0]  # shape(h,w,c)
        width = img.shape[1]

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
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

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

        return img, label_new, boxes_new


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

            self.transform = [
                augment_hsv(p=1.0),
                random_perspective(p=1.0),
                Normalize(), # 255로 나누기
                
                # RandomVFlip(p=0.5),
                # Oneof([
                #     RandomZoomIn(p=1.0, zoom_range=0.4),
                #     RandomZoomOut(p=1.0, zoom_range=0.4)
                # ], p=0.9),
                # RandomRotation(degree=5, p=0.2),

                # Oneof([
                #     brighter(brightness=0.25, p=1.0),
                #     darker(darkness=0.25, p=1.0)
                # ], p=0.2),
                
                DeNormalize(), # 255 곱하기
                Resize(input_size),
                
                ]

                
        def __call__(self, img, labels, boxes):
            for tform in self.transform:
                img, labels, boxes = tform(img, labels, boxes)

            return img, labels, boxes

    config = {'INPUT_SIZE': [300, 300]}
    aug = augmentator(config)
    
    while True:
        # img = cv2.imread('sample/000000007900.jpg')
        # label = [0]
        # boxes = [[0.2710937559604645, 0.37361112236976624, 0.12656249105930328, 0.569444477558136]]
        
        img = cv2.imread('sample/000000009000.jpg')
        label = [0]
        boxes = [[0.06406250596046448, 0.38750001788139343, 0.125, 0.4694444537162781]]
        
        img, label, boxes = aug(img, label, boxes)
        img = cv2.resize(img, (300, 300))
        h, w, _ = img.shape
        
        for box in boxes:
            x1 = int(w * (box[0] - (box[2] / 2.0)))
            x2 = int(w * (box[0] + (box[2] / 2.0)))
            y1 = int(h * (box[1] - (box[3] / 2.0)))
            y2 = int(h * (box[1] + (box[3] / 2.0)))
        
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        print(img[0, 0, :], np.max(np.max(np.max(img))), type(img), img.dtype)
        
        cv2.imshow('aug', img)
        cv2.waitKey(0)
        
    