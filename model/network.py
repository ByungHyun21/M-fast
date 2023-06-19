import torch.optim as optim

def network(cfg):
    # 학습률 설정
    lr0 = cfg['training']['lr0'] * cfg['training']['batch_size'] / 32 # batch size 64

    # SSD 모델 생성
    if cfg['network']['type'] == 'ssd':
        cfg['network']['num_classes'] += 1 # add background class
        
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
       
        
      
    return model_full, preprocess, augmentation, losses

def get_optimizer_scheduler(cfg, model):
    
    if cfg['training']['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(model_full.model.parameters(), 
                              lr=lr0, 
                              momentum=cfg['training']['optimizer']['momentum'], 
                              weight_decay=cfg['training']['optimizer']['weight_decay'])
    if cfg['training']['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(model_full.model.parameters(), 
                               lr=lr0, 
                               betas=(0.9, 0.999), 
                               weight_decay=cfg['training']['optimizer']['weight_decay'])
    
    if cfg['training']['scheduler']['type'] == 'steplr':
        def custom_scheduler(step):
            if step < cfg['training']['scheduler']['steplr'][0]:
                lr = 1 # learning_rate = lr0 * lr
            elif step < cfg['training']['scheduler']['steplr'][1]:
                lr = cfg['training']['scheduler']['gamma']
            else:
                lr = cfg['training']['scheduler']['gamma'] ** 2
            return lr
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)
        
    if cfg['training']['scheduler']['type'] == 'decaylr':
        def custom_scheduler(step):
            lr = cfg['training']['scheduler']['gamma'] ** (step / 10000)
            return lr
    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)
        
    optimizer.zero_grad()  
    return optimizer, scheduler