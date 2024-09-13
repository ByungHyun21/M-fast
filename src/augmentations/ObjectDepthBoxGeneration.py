import random
import cv2
import numpy as np
import math
import copy
import os

class ObjectDepthBoxGeneration(object):
    def __init__(self, cfg, p=1.0):
        self.cfg = cfg
        self.p = p
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, img:np.array, label):
        if random.random() > self.p:
            return img, label
        
        label['flag'].append('ObjectDepthBoxGeneration')
        
        depthbox_path = label['image_path'].replace('Image', 'DepthBox')
        depthbox_path = depthbox_path.split('.')[0] + '.png'
        if os.path.exists(depthbox_path):
            depthbox = cv2.imread(depthbox_path)
            if depthbox.shape[2] == 1:
                depthbox = cv2.cvtColor(depthbox, cv2.COLOR_GRAY2RGB)
            label['depthbox'] = depthbox
            return img, label
        
        temp_dir = f"temp/{self.cfg['dataset']}/" + label['image_path'].split('/')[-2]
        depthbox_path = temp_dir + '/' + label['image_path'].split('/')[-1].split('.')[0] + '.png'
        if not os.path.exists(f"temp/{self.cfg['dataset']}"):
            os.mkdir(f"temp/{self.cfg['dataset']}")
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        if os.path.exists(depthbox_path):
            depthbox = cv2.imread(depthbox_path)
            if depthbox.shape[2] == 1:
                depthbox = cv2.cvtColor(depthbox, cv2.COLOR_GRAY2RGB)
            label['depthbox'] = depthbox
            return img, label
        
        imh, imw = img.shape[:2]
        
        intrinsic = label['intrinsic']
        
        n_objects = len(label['objects'])
        triangles = np.array([
                [0, 1, 2],
                [0, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
                [0, 4, 7],
                [0, 7, 3],
                [1, 5, 6],
                [1, 6, 2],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6]
            ], dtype=np.int32)
        
        depthbox = np.ones((imh, imw, 3), dtype=np.uint8) * np.inf
        for obj in label['objects']:
            box2d_cx = float(obj['box2d']['cx']) * imw
            box2d_cy = float(obj['box2d']['cy']) * imh
            box2d_w = float(obj['box2d']['w']) * imw
            box2d_h = float(obj['box2d']['h']) * imh
            
            size_w = float(obj['box3d']['size']['width'])
            size_h = float(obj['box3d']['size']['height'])
            size_l = float(obj['box3d']['size']['length'])
            
            rotation = np.array(obj['box3d']['rotation']).reshape(3, 3)
            tx = obj['box3d']['translation']['x']
            ty = obj['box3d']['translation']['y']
            tz = obj['box3d']['translation']['z']
            
            if tz < 0:
                continue
            
            translation = np.array([tx, ty, tz]).reshape(3, 1)
            
            points = np.array([
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0],
                [1, 0, 0, 1, 1, 0, 0, 1]
            ], dtype=np.float32)

            points[0, :] = points[0, :] * size_l - size_l / 2
            points[1, :] = points[1, :] * size_w - size_w / 2
            points[2, :] = points[2, :] * size_h - size_h / 2

            points = rotation @ points
            points = points + translation
            
            x1, x2 = int(box2d_cx - box2d_w / 2), int(box2d_cx + box2d_w / 2)
            y1, y2 = int(box2d_cy - box2d_h / 2), int(box2d_cy + box2d_h / 2)
            for u in range(x1, x2):
                for v in range(y1, y2):
                    ray = np.array([u, v, 1]).reshape(3, 1)
                    ray = np.linalg.inv(intrinsic) @ ray
                    
                    for triangle in triangles:
                        p0 = points[:, triangle[0]].reshape(3, 1)
                        p1 = points[:, triangle[1]].reshape(3, 1)
                        p2 = points[:, triangle[2]].reshape(3, 1)
                        
                        v0 = p1 - p0
                        v1 = p2 - p1
                        v2 = p0 - p2
                        
                        #plane normal
                        a = v0[1] * v1[2] - v0[2] * v1[1]
                        b = v0[2] * v1[0] - v0[0] * v1[2]
                        c = v0[0] * v1[1] - v0[1] * v1[0]
                        plane_normal = np.array([a, b, c]).reshape(3, 1)
                        d = -a * p0[0] - b * p0[1] - c * p0[2]
                        
                        # p3d: t * ray
                        t = -d / (a * ray[0] + b * ray[1] + c * ray[2])
                        p3d = t * ray
                        
                        if t < 0:
                            continue
                        
                        depth = int(p3d[2])
                        if depth < 0:
                            continue
                        
                        v0_ = p0 - p3d
                        v1_ = p1 - p3d
                        v2_ = p2 - p3d
                        c0 = np.cross(v0.T, v0_.T)
                        c1 = np.cross(v1.T, v1_.T)
                        c2 = np.cross(v2.T, v2_.T)
                        
                        b0 = True if np.dot(c0, plane_normal) > 0 else False
                        b1 = True if np.dot(c1, plane_normal) > 0 else False
                        b2 = True if np.dot(c2, plane_normal) > 0 else False
                        
                        if (b0 == b1) and (b1 == b2):
                            depthbox[v, u] = np.minimum(depth, depthbox[v, u])
            
        depthbox[depthbox == np.inf] = 0
        label['depthbox'] = depthbox
        
        cv2.imwrite(depthbox_path, depthbox)
        
        return img, label
    