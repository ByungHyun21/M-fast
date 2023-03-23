<p align="center"><img src="./images/M-FastLogo.png" width="700px" height="200px" title="M-Fast Logo"/></p>

# M-fast
Mono AI 

### TODO
- Multi-GPU + Docker 지원
- Target에 따른 학습 파이프라인 지원(acacia, belladonna, etc.)
- Backbone에 따른 학습 파이프라인 지원(Vgg16, MobileNet-V1, MobileNet-V2, MobileNet-V3)
- Pretrained 여부에 따른 학습 파이프라인 지원(Pretrained, Not Pretrained, MOCO)
- Dataset Merge에 따른 학습 파이프라인 지원(COCO+VOC)
    * utils/load_config에서 현재 단일 데이터셋만 지원
- 다른 Dataset에 대한 평가 파이프라인 지원(COCO, VOC, CrowdHuman, Argoseye)

### 사용법

##### Nvidia-docker 설치
```bash
TODO
```

##### Dockerfile을 이용한 실행
```bash
docker build --tag mfast .

docker run -it --rm mfast \
--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
-v {M-fast 경로}:/mfast \
-v {데이터셋 경로}:/dataset \ 
-shm-size=8G
```

##### Docker 내부에서 실행
```bash
TODO
```

# Backbone
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
##### ImageNet Dataset
|Backbone|Method|Dataset|Top-1|Top-5|
|:---:|:---:|:---:|:---:|:---:|
|Vgg16|Classification|ImageNet|-|-|
|MobileNet-V1|Classification|ImageNet|-|-|
|MobileNet-V2|Classification|ImageNet|-|-|
|MobileNet-V3|Classification|ImageNet|-|-|

##### OpenImagesV7 Dataset
|Backbone|Method|Dataset|Top-1|Top-5|
|:---:|:---:|:---:|:---:|:---:|
|Vgg16|MOCO|OpenImagesV7|-|-|
|MobileNet-V1|MOCO|OpenImagesV7|-|-|
|MobileNet-V2|MOCO|OpenImagesV7|-|-|
|MobileNet-V3|MOCO|OpenImagesV7|-|-|

##### PASS Dataset
|Backbone|Method|Dataset|Top-1|Top-5|
|:---:|:---:|:---:|:---:|:---:|
|Vgg16|MOCO|PASS|-|-|
|MobileNet-V1|MOCO|PASS|-|-|
|MobileNet-V2|MOCO|PASS|-|-|
|MobileNet-V3|MOCO|PASS|-|-|
  
# Model
### 지원하는 모델
##### 2016
- [ ] YoloV1 (You Only Look Once, CVPR 2016)
- [ ] SSD (Single Shot MultiBox Detector, ECCV 2016)
  - vgg16, mobilenet-v1, mobilenet-v2, mobilenet-v3
- [ ] YOLO9000 (YOLO9000: Better, Faster, Stronger, CVPR 2017)

##### 2019
- [ ] CenterNet (Objects as Points, CVPR 2019)


### 지원하는 Dataset
- [ ] VOC2007+2012 (PASCAL VOC, 20 classes)
- [ ] COCO2017 (Common Objects in Context, 80 classes)
- [ ] Crowd Human (Crowd Human, 2 class)
- [ ] Argoseye (Argoseye, 1 class)

### 성능
##### COCO2017 Dataset
|Model|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|
|SSD|-|-|-|-|-|-|

##### VOC2007+2012 Dataset
|Model|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|
|SSD|-|-|-|-|-|-|

##### CrowdHuman Dataset
|Model|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|
|SSD|-|-|-|-|-|-|

##### Argoseye Dataset
|Model|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|
|SSD|-|-|-|-|-|-|