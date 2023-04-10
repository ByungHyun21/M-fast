def network(config, rank, istrain):
    manager = None

    if config['METHOD'] == 'ssd':
        
        from .type.ssd.ssd_full import ssd_full
        from .type.ssd.anchor import anchor_generator
        from .type.ssd.augmentator import augmentator
        from .type.ssd.preprocess import preprocessor
        from .type.ssd.loss import loss
        
        if config['TYPE'] == 'vgg16':    
            from .type.ssd.ssd_vgg16 import ssd_vgg16
            model = ssd_vgg16(config)
        if config['TYPE'] == 'mobilenet_v2':
            from .type.ssd.ssd_mobilenet_v2 import ssd_mobilenet_v2
            model = ssd_mobilenet_v2(config)
        
        anchor = anchor_generator(config)
        
        model_full = ssd_full(config, model, anchor)
        preprocessor = preprocessor(config, anchor)
        augmentator = augmentator(config)
        loss = loss(config)
        
    return model_full, preprocessor, augmentator, loss