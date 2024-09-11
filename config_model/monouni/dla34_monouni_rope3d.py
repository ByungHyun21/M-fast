import json

cfg = dict()

cfg['model'] = 'monouni'
cfg['backbone'] = 'dla34'
cfg['dataset'] = 'Rope3D_2100'
cfg['classes'] = ["pedestrian", "car", "bus", "truck", "van", "cyclist"]
cfg['num_classes'] = 4
cfg['task'] = ['box2d', 'box3d']
cfg['metric'] = []
cfg['input_shape'] = [960, 512]
cfg['batch_size'] = 8
cfg['end_epoch'] = 300
cfg['num_workers'] = 4
cfg['lr0'] = 0.036
cfg['backbone_freeze'] = False
cfg['neck_freeze'] = False
cfg['head_freeze'] = False
cfg['backbone_init_weight'] = 'torchvision'
cfg['model_init_weight'] = None

cfg['optimizer'] = dict()
cfg['optimizer']['sgd'] = dict()
cfg['optimizer']['sgd']['weight_decay'] = 0.4e-5
cfg['optimizer']['sgd']['momentum'] = 0.9
cfg['scheduler'] = dict()
cfg['scheduler']['cosineannealinglr'] = dict()
cfg['scheduler']['cosineannealinglr']['T_max'] = 300
cfg['scheduler']['cosineannealinglr']['eta_min'] = 1e-5
cfg['scheduler']['cosineannealinglr']['last_epoch'] = -1

cfg['loss'] = dict()
cfg['loss']['alpha'] = dict()
cfg['loss']['alpha']['class'] = 1.0
cfg['loss']['alpha']['location'] = 1.0

# cfg['metric'] = list()
# cfg['metric'].append(['mAP', '101point'])

cfg['info'] = dict()
# W, H, L mu
cfg['info']['avg_size'] = [[0.47977883671931515, 1.603604342354993, 0.46636183661653013],
                           [1.7026277174690747, 1.2947722138378017, 4.233722115187554],
                           [2.5578943573900097, 2.9520296547207043, 10.664581956005929],
                           [2.391652094214645, 2.6833085765071707, 6.597212293728038],
                           [1.7653359832248707, 1.7347598715112018, 4.639641923499641],
                           [0.4899444975525569, 1.3298419847519023, 1.5058671495932674]]


if __name__ == "__main__":
    import os 
    
    file_name = __file__.replace('.py', '.json')
    
    if os.path.exists(file_name):
        os.remove(file_name)
    
    #save json
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)