from model.augmentations import *

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

        self.transform_train = [
            # photometric
            augment_hsv(hgain=hsv_hgain, sgain=hsv_sgain, vgain=hsv_vgain, p=hsv_prob),

            # resize
            Resize(input_size),

            # geometric
            mosaic(canvas_range=mosaic_canvas_range, p=mosaic_prob),
            random_perspective(degree=perspective_degree, translate=perspective_translate, scale=perspective_scale, shear=perspective_shear, perspective=perspective_perspective, p=perspective_prob),

            # resize
            Resize(input_size),
            RandomVFlip(p=0.5),
            ]

        self.transform_valid = [
            Resize(input_size),
            RandomVFlip(p=0.5),
            ]
            
    def __call__(self, img, labels, boxes, istrain):
        if istrain:
            for tform in self.transform_train:
                img, labels, boxes = tform(img, labels, boxes)
        else:
            for tform in self.transform_valid:
                img, labels, boxes = tform(img, labels, boxes)
        
        return img, labels, boxes