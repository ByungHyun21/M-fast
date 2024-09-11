from src.augmentation import *

class augmentator(object):
    def __init__(self, cfg):
        self.transform_train = [
            ObjectDepthBoxGeneration(p=1.0),
            FlipWithOpticalCenter(p=0.5),
            
            # resize
            Resize(cfg['input_shape']),
            ImageNormalization(p=1.0),
            ]

        self.transform_valid = [
            RandomVFlip(p=0.5),
            Resize(cfg['input_shape']),
            ImageNormalization(p=1.0),
            ]
            
    def __call__(self, img, labels, istrain):
        labels['flag'] = []
        if istrain:
            for tform in self.transform_train:
                img, labels = tform(img, labels)
        else:
            for tform in self.transform_valid:
                img, labels = tform(img, labels)
        
        return img, labels