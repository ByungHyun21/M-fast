import cv2
import numpy as np
import random


class CropWithOpticalCenter(object):
    def __init__(self, p=0.5):
        self.p = p
        
    #@numba.jit(nopython=True, cache=True)
    def __call__(self, data, label):
        if random.random() > self.p:
            return data, label
        label['flag'].append('FlipLR_mono')
        
        data = cv2.flip(data, 1)

        imh, imw, _ = data.shape
    
        #* 검증 완료
        if 'intrinsic' in label: 
            # camera principal point x flip
            label['intrinsic'][0, 2] = imw - label['intrinsic'][0, 2] 

        for obj in label['object']:
            #TODO: 구현 필요
            if 'polygon' in obj:
                for poly in obj['polygon']:
                    for point in poly:
                        point[0] = 1 - point[0]
                
            #*: 검증 완료
            if 'box2d' in obj and 'polygon' not in obj: 
                obj['box2d']['cx'] = 1 - obj['box2d']['cx']
        
        return data, label
    
if __name__ == "__main__":
    import os, json
    import cv2
    import time

    sample_dir = 'sample/KITTI_Subset/StereoL'
    _label_list = os.listdir(sample_dir)
    label_list = []
    for label in _label_list:
        if label.endswith('.json'):
            label_list.append(label)
    label_list.sort()

    cnt = 0

    aug = FlipLR_mono(p=0.5)
    
    front_color = (0, 0, 128)
    side_color = (0, 128, 0)
    back_color = (128, 0, 0)

    while True:
        cnt = (cnt + 1) % len(label_list)
        print('frame: ', cnt)
        #label
        with open(os.path.join(sample_dir, label_list[cnt]), 'r') as f:
            label = json.load(f)
        
        image = cv2.imread(os.path.join(sample_dir, label_list[cnt][:-5] + '.jpg'))

        # data keys
        #   - 'StereoL': np.array
        # label keys
        #   - 'StereoL': dict ('extrinsic_rotation', 'extrinsic_translation', 'intrinsic', 'object')
        #   - 'data_type': 'image'

        data = {'StereoL': image}
        label = {'StereoL': label}
        label['StereoL']['data_type'] = 'image'
        label['StereoL']['flag'] = []

        aug_start = time.time()
        flag = []
        data, label = aug(data=data, label=label)
        aug_end = time.time()

        image = data['StereoL']
        h, w, _ = image.shape

        objects = label['StereoL']['object']
        for obj in objects:
            # if 'box2d' in obj:
            #     box2d = obj['box2d']
            #     cx = int(box2d['cx'] * w)
            #     cy = int(box2d['cy'] * h)
            #     w = int(box2d['w'] * w)
            #     h = int(box2d['h'] * h)

            #     cv2.rectangle(image, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (0, 0, 255), 2)

            if 'polygon' in obj:
                polygon = obj['polygon']
                for poly in polygon:
                    for i in range(len(poly)):
                        cv2.circle(image, (int(poly[i][0]), int(poly[i][1])), 3, (0, 255, 0), -1)
                        cv2.line(image, (int(poly[i][0]), int(poly[i][1])), (int(poly[(i+1)%len(poly)][0]), int(poly[(i+1)%len(poly)][1])), (0, 255, 0), 2)

            if 'keypoint' in obj:
                keypoint = obj['keypoint']
                for key in keypoint:
                    cv2.circle(image, (int(keypoint[key][0]), int(keypoint[key][1])), 3, (255, 0, 0), -1)

            if 'box3d' in obj:
                intrinsic = np.array(label['StereoL']['intrinsic'], dtype=np.float32).reshape(3, 3)
                
                default_box = np.array([
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, -0.5, -0.5],
                    [-0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [-0.5, -0.5, -0.5]
                ])
                front_edges = np.array([
                    [0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 3]
                ])
                
                side_edges = np.array([
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7]
                ])
                
                back_edges = np.array([
                    [4, 5],
                    [4, 6],
                    [5, 7],
                    [6, 7]
                ])

                dimension = [obj['box3d']['size']['length'], obj['box3d']['size']['width'], obj['box3d']['size']['height']]
                translation = np.array(obj['box3d']['translation'], dtype=np.float32).reshape(3, 1)
                rotation = np.array(obj['box3d']['rotation'], dtype=np.float32).reshape(3, 3)

                box = default_box * dimension
                box = np.matmul(rotation, box.T).T + translation.T
                box = np.matmul(intrinsic, box.T).T
                box = box[:, :2] / box[:, 2:]

                for edge in front_edges:
                    cv2.line(image, (int(box[edge[0]][0]), int(box[edge[0]][1])), (int(box[edge[1]][0]), int(box[edge[1]][1])), front_color, 2)
                for edge in side_edges:
                    cv2.line(image, (int(box[edge[0]][0]), int(box[edge[0]][1])), (int(box[edge[1]][0]), int(box[edge[1]][1])), side_color, 2)
                for edge in back_edges:
                    cv2.line(image, (int(box[edge[0]][0]), int(box[edge[0]][1])), (int(box[edge[1]][0]), int(box[edge[1]][1])), back_color, 2)
                
        
        cv2.imshow('image', image)

        n_objects = len(label['StereoL']['object'])
        print(f"Augmentation time: {aug_end - aug_start:.4f} sec, # of objects: {n_objects}")

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        


