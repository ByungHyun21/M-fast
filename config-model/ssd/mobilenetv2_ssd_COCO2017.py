import json

cfg = dict()
cfg['test'] = False

cfg['dataset'] = 'COCO2017'
cfg['dataset_root'] = '/mnt/dataset'

cfg['network'] = dict()
cfg['network']['classes'] = ["hot dog", "dog", "potted plant", "tv", "bird", "cat", "horse", "sheep", "cow", "bottle", "couch", "chair", "dining table", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "boat", "person", "stop sign", "umbrella", "tie", "sports ball", "sandwich", "bed", "cell phone", "refrigerator", "clock", "toothbrush", "truck", "traffic light", "fire hydrant", "parking meter", "bench", "elephant", "giraffe", "frisbee", "skis", "snowboard", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "orange", "broccoli", "carrot", "pizza", "donut", "cake", "toilet", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster", "sink", "book", "vase", "scissors", "teddy bear", "hair drier", "backpack", "handbag", "suitcase", "zebra", "bear"]
cfg['network']['task'] = ['box2d']
cfg['network']['type'] = 'ssd'
cfg['network']['input_size'] = [300, 300]
cfg['network']['mean'] = [0.485, 0.456, 0.406]
cfg['network']['std'] = [0.229, 0.224, 0.225]

cfg['network']['backbone'] = 'mobilenet_v2'
cfg['network']['neck'] = 'mobilenet_v2_neck'
cfg['network']['head'] = 'mobilenet_v2_head'
cfg['network']['backbone_freeze'] = False
cfg['network']['neck_freeze'] = False
cfg['network']['head_freeze'] = False
cfg['network']['init_weight'] = 'torchvision'


cfg['anchor'] = dict()
cfg['anchor']['num_anchor'] = [6, 6, 6, 6, 6, 6]
cfg['anchor']['scales'] = [0.1, 0.35, 0.5, 0.65, 0.8, 0.95]
cfg['anchor']['grid_w'] = [19, 10, 5, 3, 2, 1]
cfg['anchor']['grid_h'] = [19, 10, 5, 3, 2, 1]

cfg['training'] = dict()
cfg['training']['batch_size'] = 192
cfg['training']['end_epoch'] = 600
cfg['training']['num_workers'] = 8
cfg['training']['lr0'] = 0.015
cfg['training']['optimizer'] = dict()
cfg['training']['optimizer']['sgd'] = dict()
cfg['training']['optimizer']['sgd']['type'] = 'sgd'
cfg['training']['optimizer']['sgd']['weight_decay'] = 0.4e-5
cfg['training']['optimizer']['sgd']['momentum'] = 0.9
cfg['training']['scheduler'] = dict()
cfg['training']['scheduler']['cosineannealinglr'] = dict()
cfg['training']['scheduler']['cosineannealinglr']['T_max'] = 600
cfg['training']['scheduler']['cosineannealinglr']['eta_min'] = 1e-5
cfg['training']['scheduler']['cosineannealinglr']['last_epoch'] = -1

cfg['loss'] = dict()
cfg['loss']['type'] = ['Total', 'Class', 'Location']
cfg['loss']['alpha'] = dict()
cfg['loss']['alpha']['class'] = 1.0
cfg['loss']['alpha']['location'] = 1.0

cfg['metric'] = list()
cfg['metric'].append(['mAP', '101point'])

cfg['nms'] = dict()
cfg['nms']['topk'] = 200
cfg['nms']['iou_threshold'] = 0.5

cfg['augmentation'] = dict()
cfg['augmentation']['hsv'] = dict()
cfg['augmentation']['hsv']['prob'] = 1.0
cfg['augmentation']['hsv']['hgain'] = 0.015
cfg['augmentation']['hsv']['sgain'] = 0.7
cfg['augmentation']['hsv']['vgain'] = 0.4

cfg['augmentation']['mosaic'] = dict()
cfg['augmentation']['mosaic']['prob'] = 0
cfg['augmentation']['mosaic']['canvas_range'] = 1.0

cfg['augmentation']['flip'] = dict()
cfg['augmentation']['flip']['prob'] = 0.5

cfg['augmentation']['crop'] = dict()
cfg['augmentation']['crop']['prob'] = 1.0
cfg['augmentation']['crop']['min_overlap'] = 0.3

cfg['augmentation']['perspective'] = dict()
cfg['augmentation']['perspective']['prob'] = 0
cfg['augmentation']['perspective']['scale'] = 0.1
cfg['augmentation']['perspective']['degree'] = 10.0
cfg['augmentation']['perspective']['translate'] = 0.1
cfg['augmentation']['perspective']['shear'] = 5.0
cfg['augmentation']['perspective']['perspective'] = 0.0

cfg['augmentation']['zoomout'] = dict()
cfg['augmentation']['zoomout']['prob'] = 0.5
cfg['augmentation']['zoomout']['max_ratio'] = 4.0

if __name__ == "__main__":
    import os 
    
    file_name = __file__.replace('.py', '.json')
    
    if os.path.exists(file_name):
        os.remove(file_name)
    
    #save json
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)
        