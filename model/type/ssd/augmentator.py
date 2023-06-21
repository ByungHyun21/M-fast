from model.augmentations import *

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
            # geometric
            RandomZoomOut(max_scale=random_zoomout_cfg['max_ratio'],
                          mean=mean,
                          p=random_zoomout_cfg['prob']),
            RandomCrop(min_overlap=random_crop_cfg['min_overlap'], 
                       p=random_crop_cfg['prob']),
            mosaic(canvas_range=mosaic_cfg['canvas_range'], 
                   p=mosaic_cfg['prob']),
            random_perspective(degree=perspective_cfg['degree'], 
                               translate=perspective_cfg['translate'], 
                               scale=perspective_cfg['scale'], 
                               shear=perspective_cfg['shear'], 
                               perspective=perspective_cfg['perspective'], 
                               p=perspective_cfg['prob']),
            # resize
            Resize(input_size),
            
            # resize
            RandomVFlip(p=flip_cfg['prob']),
            
            # photometric
            augment_hsv(hgain=hsv_cfg['hgain'], 
                        sgain=hsv_cfg['sgain'], 
                        vgain=hsv_cfg['vgain'], 
                        p=hsv_cfg['prob']),
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