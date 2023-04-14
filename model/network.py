import torch.optim as optim

def network(config, rank, istrain):
    manager = None

    lr0 = config['LR'] * config['BATCH_SIZE_MULTI_GPU'] / 32 # batch size 64

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
        preprocess = preprocessor(config, anchor)
        augmentation = augmentator(config)
        losses = loss(config)
        
        biases = list()
        not_biases = list()
        for param_name, param in model_full.model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)

        if 'vgg' in config['TYPE']:
            # optimizer = optim.SGD(model_full.model.parameters(), lr=lr0, momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
            # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr0}, {'params': not_biases}],
            #                             lr=lr0, momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
            optimizer = optim.Adam(model_full.model.parameters(), lr=lr0/10.0, betas=(0.9, 0.999), weight_decay=config['WEIGHT_DECAY'])

            def custom_scheduler(step):
                if step < config['STEPLR'][0]:
                    lr = 1 # learning_rate = lr0 * lr
                elif step < config['STEPLR'][1]:
                    lr = 0.1
                else:
                    lr = 0.01
                return lr
            
            # def custom_scheduler(step):
            #     lr = (0.92) ** (step // 10000)
            #     return lr

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)

        elif 'mobilenet' in config['TYPE']:
            # optimizer = optim.SGD(model_full.model.parameters(), lr=lr0, momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
            optimizer = optim.Adam(model_full.model.parameters(), lr=lr0/10.0, betas=(0.9, 0.999), weight_decay=config['WEIGHT_DECAY'])
            def custom_scheduler(step):
                if step < config['STEPLR'][0]:
                    lr = 1 # learning_rate = lr0 * lr
                elif step < config['STEPLR'][1]:
                    lr = 0.1
                else:
                    lr = 0.01
                return lr
        
            # def custom_scheduler(step):
            #     lr = (0.92) ** (step // 10000)
            #     return lr

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_scheduler)

    optimizer.zero_grad()    
    return model_full, preprocess, augmentation, losses, optimizer, scheduler