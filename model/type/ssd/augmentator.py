from model.augmentations import *

class augmentator(object):
    def __init__(self, config):
        input_size = config['INPUT_SIZE']

        self.transform_train = [
            augment_hsv(p=1.0),
            # random_perspective(p=1.0),
            RandomVFlip(p=0.5),
            Resize(input_size),
            ]

        self.transform_valid = [
            RandomVFlip(p=0.5),
            Resize(input_size),
            ]
            
    def __call__(self, img, labels, boxes, istrain):
        if istrain:
            for tform in self.transform_train:
                img, labels, boxes = tform(img, labels, boxes)
        else:
            for tform in self.transform_valid:
                img, labels, boxes = tform(img, labels, boxes)
        
        return img, labels, boxes