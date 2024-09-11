from src.augmentation import *

class augmentator(object):
    def __init__(self, cfg):
        input_size = cfg['input_shape']

        hsv_cfg = cfg['augmentation']['hsv']
        flip_cfg = cfg['augmentation']['flip']

        mean = cfg['mean']
        std = cfg['std']

        self.transform_train = [
            RandomVFlip(p=flip_cfg['prob']),
            
            # photometric
            AugmentHSV(hgain=hsv_cfg['hgain'], 
                        sgain=hsv_cfg['sgain'], 
                        vgain=hsv_cfg['vgain'], 
                        p=hsv_cfg['prob']),
            
            
            
            # resize
            Resize(input_size),
            ]

        self.transform_valid = [
            RandomVFlip(p=0.5),
            Resize(input_size),
            ]
            
    def __call__(self, img, labels, istrain):
        if istrain:
            for tform in self.transform_train:
                img, labels = tform(img, labels)
        else:
            for tform in self.transform_valid:
                img, labels = tform(img, labels)
        
        return img, labels