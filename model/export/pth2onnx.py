import argparse
import os
import yaml

import torch

from model.od_dataloader import get_dataloader
from model.network import network
from model.od_ssd import ssd_post

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/mv2.yaml', help='plan has training info like epoch, batch_size, ... etc')
opt = parser.parse_args()
with open(opt.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# # # # #
# torch, CUDA, gpu Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def train(ddp_rank):
        # 모델 불러오기
        # model.load_state_dict(torch.load(PATH))
        # model.eval()
    model, _, _, _, _, _ = network(config, ddp_rank)
    post_model = ssd_post(model)
    post_model.to('cuda')
    post_model.eval()
    torch.onnx.export(post_model, torch.rand(1, 3, 300, 300), 'model.onnx', opset_version=10, export_params=True)
    print('done')


if __name__ == '__main__':
    train(0)
    # mp.spawn(train, nprocs=torch.cuda.device_count(), join=True)