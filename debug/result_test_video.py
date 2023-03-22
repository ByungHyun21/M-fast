import argparse
import os, yaml, cv2, time
import numpy as np
from tqdm import tqdm

import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.network_manager import network_manager
from model.od_ssd import ssd_post

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/mv2_coco_80.yaml', help='plan has training info like epoch, batch_size, ... etc')
parser.add_argument('--gpu', type=str, default='0', help='GPUs eg. 0 or 0,1 or 2,3,4,5')
opt = parser.parse_args()
with open(opt.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# # # # #
# torch, CUDA, gpu Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
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
    
    test_dir = os.path.join(config['DATASET_PATH'], 'argoseye', 'test_video')
    sub_dir = os.listdir(test_dir)
    sub_dir.sort()
    
    for sd in sub_dir:
        imlist = os.listdir(os.path.join(test_dir, sd))
        imlist.sort()
        
        video_file_name = os.path.join('runs', config['PROJECT'], sd + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        vw = 640
        vh = 320
        
        video = cv2.VideoWriter(video_file_name, fourcc, 30, (vw, vh))
        
        for im in tqdm(imlist, total=len(imlist)):
            img_org = cv2.imread(os.path.join(test_dir, sd, im))
            img = cv2.resize(img_org, (300, 300))
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0).astype(float)
        
            detections = post_model(torch.from_numpy(img).to(device).float())
            detections = detections.detach().cpu().numpy()
            
            img_org = cv2.resize(img_org, (vw, vh))
            
            for box in detections:
                cls = np.argmax(box[1:-4])
                score = box[1] # only person
                if score < 0.5:
                    continue
                
                txt = str(int(cls)) + ', ' +  str(int(score * 100) / 100)
                
                x1 = int(vw * box[-4])
                y1 = int(vh * box[-3])
                x2 = int(vw * box[-2])
                y2 = int(vh * box[-1])
                
                color = (0, 255, 0)
                cv2.rectangle(img_org, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img_org, txt, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                
            video.write(img_org)
        video.release()
    


if __name__ == '__main__':
    train(0)