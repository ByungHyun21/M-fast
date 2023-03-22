import torch
from model.od_ssd import ssd, ssd_preprocessor, ssd_augmentator, ssd_loss, ssd_metric, ssd_manager

def network_manager(config, ddp_rank, istrain):
    manager = None

    # # # # #
    # get Model
    if config['method'] == 'ssd':
        model = ssd(config, ddp_rank)
        anchor = model.get_anchor()
        preprocessor = ssd_preprocessor(config, anchor)
        augmentator = ssd_augmentator(config)
        
        loss = ssd_loss()
        metric = ssd_metric()
        if istrain:
            manager = ssd_manager(config, ddp_rank)
        
    return model, preprocessor, augmentator, loss, metric, manager