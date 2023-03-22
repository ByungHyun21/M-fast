def network_manager(config, ddp_rank, istrain):
    manager = None

    # # # # #
    # get Model
    if config['method'] == 'ssd':
        from .type.ssd.preprocess import preprocess
        from .type.ssd.postprocess import postprocess
        from .type.ssd.model import model
        from .type.ssd.loss import loss
        from .type.ssd.metric import metric
        
        model = ssd(config, ddp_rank)
        anchor = model.get_anchor()
        preprocessor = ssd_preprocessor(config, anchor)
        augmentator = ssd_augmentator(config)
        
        loss = ssd_loss()
        metric = ssd_metric()
        if istrain:
            manager = ssd_manager(config, ddp_rank)
        
    return model, preprocessor, augmentator, loss, metric, manager