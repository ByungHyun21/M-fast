<p align="center"><img src="./images/M-FastLogo.png" width="700px" height="200px" title="M-Fast Logo"/></p>

# M-fast 🏃‍♂️🏃🏃‍♀️
Mono AI 

## :small_blue_diamond: TODO 
- One-node, Multi-GPU 지원
- Multi-node, Multi-GPU 지원
- Target에 따른 학습 파이프라인 지원(acacia, belladonna, etc.)
- Pretrained 여부에 따른 학습 파이프라인 지원(Pretrained, Not Pretrained, MOCO)
- Dataset Merge에 따른 학습 파이프라인 지원(COCO+VOC)
    * utils/load_config에서 현재 단일 데이터셋만 지원
- 다른 Dataset에 대한 평가 파이프라인 지원(COCO, VOC, CrowdHuman, Argoseye)

## :one: 사용법

### :small_blue_diamond: Nvidia-docker 설치
```bash
TODO
```

### :small_blue_diamond: Dockerfile을 이용한 실행
```bash
docker build --tag mfast .

docker run -it --rm mfast \
--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
-v {M-fast 경로}:/mfast \
-v {데이터셋 경로}:/dataset \ 
-shm-size=8G
```

예시
```
docker run -it --rm --gpus=all -v C:\M-fast:/mfast -v C:\dataset:/dataset --shm-size=8g mfast
```

### Docker 내부에서 실행
#### Single Node Single GPU 
  - COCO 
```bash
python train_model_single.py --config {config 경로} --coco 
```
  - VOC
```bash
python train_model_single.py --config {config 경로} --voc
```
  - CrowdHuman
```bash
python train_model_single.py --config {config 경로} --crowdhuman
```
  - Argoseye
```bash
python train_model_single.py --config {config 경로} --argoseye
```

#### Single Node Multi GPU
  - COCO 
```bash
python train_model.py --config {config 경로} --coco
```
  - VOC
```bash
python train_model.py --config {config 경로} --voc
```
  - CrowdHuman
```bash
python train_model.py --config {config 경로} --crowdhuman
```
  - Argoseye
```bash
python train_model.py --config {config 경로} --argoseye
```

## :two: Backbone
### 지원하는 모델
- [ ] Vgg16
- [ ] MobileNet-V1
- [ ] MobileNet-V2
- [ ] MobileNet-V3

### 지원하는 Dataset
- [ ] ImageNet
- [ ] OpenImagesV7
- [ ] PASS

### 성능
#### :radio_button: ImageNet Dataset
|Backbone|Method|Dataset|Top-1|Top-5|
|:---:|:---:|:---:|:---:|:---:|
|Vgg16|Classification|ImageNet|-|-|
|MobileNet-V1|Classification|ImageNet|-|-|
|MobileNet-V2|Classification|ImageNet|-|-|
|MobileNet-V3|Classification|ImageNet|-|-|

#### :radio_button: OpenImagesV7 Dataset
|Backbone|Method|Dataset|Top-1|Top-5|
|:---:|:---:|:---:|:---:|:---:|
|Vgg16|MOCO|OpenImagesV7|-|-|
|MobileNet-V1|MOCO|OpenImagesV7|-|-|
|MobileNet-V2|MOCO|OpenImagesV7|-|-|
|MobileNet-V3|MOCO|OpenImagesV7|-|-|

#### :radio_button: PASS Dataset
|Backbone|Method|Dataset|Top-1|Top-5|
|:---:|:---:|:---:|:---:|:---:|
|Vgg16|MOCO|PASS|-|-|
|MobileNet-V1|MOCO|PASS|-|-|
|MobileNet-V2|MOCO|PASS|-|-|
|MobileNet-V3|MOCO|PASS|-|-|
  
## :three: Model
### 지원하는 모델
#### 2016
- [ ] YoloV1 (You Only Look Once, CVPR 2016)
- [ ] SSD (Single Shot MultiBox Detector, ECCV 2016)
  - vgg16, mobilenet-v1, mobilenet-v2, mobilenet-v3
- [ ] YOLO9000 (YOLO9000: Better, Faster, Stronger, CVPR 2017)

#### 2019
- [ ] CenterNet (Objects as Points, CVPR 2019)


### 지원하는 Dataset
- [ ] VOC2007+2012 (PASCAL VOC, 20 classes)
- [ ] COCO2017 (Common Objects in Context, 80 classes)
- [ ] Crowd Human (Crowd Human, 2 class)
- [ ] Argoseye (Argoseye, 1 class)

### 성능
#### :radio_button: COCO2017 Dataset
|Model|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|
|SSD|-|-|-|-|-|-|

#### :radio_button: VOC2007+2012 Dataset
|Model|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|
|SSD|-|-|-|-|-|-|

#### :radio_button: CrowdHuman Dataset
|Model|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|
|SSD|-|-|-|-|-|-|

#### :radio_button: Argoseye Dataset
|Model|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|
|SSD|-|-|-|-|-|-|