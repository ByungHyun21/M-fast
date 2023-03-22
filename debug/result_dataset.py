import argparse
import os, yaml, cv2, time
import numpy as np

import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.network_manager import network_manager
from model.od_ssd import ssd_post

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/mv2_argoseye_0005.yaml', help='plan has training info like epoch, batch_size, ... etc')
opt = parser.parse_args()
with open(opt.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# # # # #
# torch, CUDA, gpu Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    device = 'cpu'

def train(ddp_rank):
    model, _, _, _, _, _ = network_manager(config, ddp_rank, istrain=False)
    
    # 모델 불러오기
    model.load_state_dict(torch.load('runs/' + config['PROJECT'] + '/' + config['PROJECT'] + '_train.pt'))
    model.eval()
        
    post_model = ssd_post(model)
    post_model.to('cuda')
    post_model.eval()
    
    vc = cv2.VideoCapture(0)
    
    while True:
        ret, img_org = vc.read()
        if ret == False:
            print("frame is empty!")
            continue

        img = cv2.resize(img_org, (300, 300))
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(float)
        
        st = time.time()
        detections = post_model(torch.from_numpy(img).to(device).float())
        et = time.time()
        print(f"time: {(et - st):.2f} FPS: {1/(et - st + 1e-5):.2f}")
        
        img_org = cv2.resize(img_org, (600, 600))
        
        for box in detections:
            if box[1] < 0.5:
                continue
            
            txt = str(int(box[1].detach().cpu().numpy() * 100) / 100)
            
            x1 = int(600 * box[2])
            y1 = int(600 * box[3])
            x2 = int(600 * box[4])
            y2 = int(600 * box[5])
            
            color = (0, 255, 0)
            cv2.rectangle(img_org, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img_org, txt, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
            
        cv2.imshow('result', img_org)
        cv2.waitKey(1)
    


if __name__ == '__main__':
    train(0)
    # mp.spawn(train, nprocs=torch.cuda.device_count(), join=True)