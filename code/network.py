import torch.optim as optim

def network(cfg):
    # 학습률 설정
    lr0 = cfg['training']['lr0'] * cfg['training']['batch_size'] / 192 # batch size 64

    # SSD 모델 생성
    if cfg['network']['type'] == 'ssd':
        cfg['network']['classes'].insert(0, 'background') # add background class
        cfg['network']['num_classes'] = len(cfg['network']['classes'])
        
        from .type.ssd.ssd_full import ssd_full
        from .type.ssd.anchor import anchor_generator
        from .type.ssd.augmentator import augmentator
        from .type.ssd.preprocess import preprocessor
        from .type.ssd.loss import loss
        
        if cfg['network']['backbone'] == 'vgg16_bn':    
            from .type.ssd.ssd_vgg16_bn import ssd_vgg16_bn
            model = ssd_vgg16_bn(cfg)
        if cfg['network']['backbone'] == 'mobilenet_v2':
            from .type.ssd.ssd_mobilenet_v2 import ssd_mobilenet_v2
            model = ssd_mobilenet_v2(cfg)
        
        anchor = anchor_generator(cfg)
        
        model_full = ssd_full(cfg, model, anchor)
        preprocess = preprocessor(cfg, anchor)
        augmentation = augmentator(cfg)
        losses = loss(cfg)
     
    # YOLO 모델 생성
    
    # CenterNet 모델 생성
       
        
    if 'sgd' in cfg['training']['optimizer']:
        sgd = cfg['training']['optimizer']['sgd']
        optimizer = optim.SGD(model_full.model.parameters(), 
                              lr=lr0, 
                              momentum=sgd['momentum'], 
                              weight_decay=sgd['weight_decay'])
    if 'adam' in cfg['training']['optimizer']:
        adam = cfg['training']['optimizer']['adam']
        optimizer = optim.Adam(model_full.model.parameters(), 
                               lr=lr0, 
                               betas=(adam['beta1'], adam['beta2']), 
                               weight_decay=adam['weight_decay'])
    
    if 'steplr' in cfg['training']['scheduler']:
        steplr = cfg['training']['scheduler']['steplr']
        
        def custom_scheduler(step):
            if step < steplr['epochs'][0]:
                lr = 1 # learning_rate = lr0 * lr
            elif step < steplr['epochs'][1]:
                lr = steplr['gamma']
            else:
                lr = steplr['gamma'] ** 2
            return lr
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)
        
    if 'decaylr' in cfg['training']['scheduler']:
        def custom_scheduler(step):
            lr = cfg['training']['scheduler']['gamma'] ** (step / 10000)
            return lr
    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)
        
    if 'cosineannealinglr' in cfg['training']['scheduler']:
        cosineannealinglr = cfg['training']['scheduler']['cosineannealinglr']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         T_max=cosineannealinglr['T_max'], 
                                                         eta_min=cosineannealinglr['eta_min'],
                                                         last_epoch=cosineannealinglr['last_epoch'])
        
    optimizer.zero_grad()    
    return model_full, preprocess, augmentation, losses, optimizer, scheduler

