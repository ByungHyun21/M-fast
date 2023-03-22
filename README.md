<p align="center"><img src="./images/M-FastLogo.png" width="700px" height="200px" title="M-Fast Logo"/></p>

# M-fast
Mono AI 

### TODO
- 코드 작성 
- Multi-GPU + Docker 지원

### 사용법

##### Dockerfile을 이용한 실행
```bash
docker build --tag M-fast .

docker run -it --rm M-fast \
-v {M-fast 경로}:/M-fast \
-v {데이터셋 경로}:/dataset
```

##### Docker 내부에서 실행
```bash
python 
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
- [ ] YoloV1
- [ ] CenterNet
- [ ] SSD

### 지원하는 Dataset
- [ ] VOC2007+2012
- [ ] COCO
- [ ] Argoseye

### 성능
##### COCO Dataset
|Model|Dataset|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|COCO|-|-|-|-|-|-|
|SSD|COCO|-|-|-|-|-|-|

##### VOC2007+2012 Dataset
|Model|Dataset|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|VOC2007+2012|-|-|-|-|-|-|
|SSD|VOC2007+2012|-|-|-|-|-|-|

##### Argoseye Dataset
|Model|Dataset|AP|AP50|AP75|APs|APm|APl|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|Argoseye|-|-|-|-|-|-|
|SSD|Argoseye|-|-|-|-|-|-|