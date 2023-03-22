import yaml

class AverageMeter:
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        
    def get_data(self, data):
        self.cnt += 1
        self.sum += data
    
    def get_mean(self):
        return self.sum / self.cnt

    def reset(self):
        self.cnt = 0
        self.sum = 0
        
        
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    return config