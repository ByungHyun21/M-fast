import json

cfg = dict()

cfg['model'] = 'monouni'
cfg['backbone'] = 'dla34'
cfg['dataset'] = 'Rope3D_cits'
cfg['classes'] = ["person"]
cfg['num_classes'] = 1
cfg['task'] = ['box2d', 'box3d']
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


if __name__ == "__main__":
    import os 
    
    file_name = __file__.replace('.py', '.json')
    
    if os.path.exists(file_name):
        os.remove(file_name)
    
    #save json
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)