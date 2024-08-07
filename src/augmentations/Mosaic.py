import random
import cv2
import numpy as np
import math
import copy

class Mosaic(object):
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
