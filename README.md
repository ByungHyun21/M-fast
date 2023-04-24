<p align="center"><img src="./images/M-FastLogo.png" width="700px" height="200px" title="M-Fast Logo"/></p>

# M-fast 🏃‍♂️🏃🏃‍♀️
Mono AI 

## :small_blue_diamond: TODO 
- Multi-node, Multi-GPU 지원
- Target에 따른 학습 파이프라인 지원(acacia, belladonna, etc.)
- Pretrained 여부에 따른 학습 파이프라인 지원(Pretrained, Not Pretrained, MOCO)
- Dataset Merge에 따른 학습 파이프라인 지원(COCO+VOC)
    * utils/load_config에서 현재 단일 데이터셋만 지원
- 효율적인 mAP 계산 지원

## :one: 사용법

### :small_blue_diamond: Anaconda를 이용한 실행
```bash
conda create -n {env_name} python=3.9
conda activate {env_name}
```

```bash
(env_name) conda install -c conda-forge wandb tqdm opencv -y
(env_name) conda install pyyaml
```


### :small_blue_diamond: Docker를를 이용한 실행

#### ::radio_button:: Nvidia-docker 설치
```bash
TODO
```

#### ::radio_button:: Dockerfile을 이용한 실행
```bash
docker build --tag mfast .

docker run -it --rm \
--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
-v {M-fast 경로}:/M-fast \
-v {데이터셋 경로}:/dataset \ 
--shm-size=8G \ 
--network host \
mfast 

wandb login {API_KEY}
```

예시
```
docker run -it --rm --gpus=all -v C:\M-fast:/M-fast -v C:\dataset:/dataset --shm-size=8g --network host mfast
wandb login xxxxxxxxxxxxxxx
```

#### ::radio_button:: Docker 내부에서 실행
##### :radio_button: Single Node Single GPU 
* 인자로 '--coco', '--voc', '--crowdhuman', '--argoseye' 입력시 해당 데이터셋을 사용함 
```bash
python train_model_single.py --config {config 경로} --coco 
```

##### :radio_button: Single Node Multi GPU
* 인자로 '--coco', '--voc', '--crowdhuman', '--argoseye' 입력시 해당 데이터셋을 사용함 
```bash
python train_model.py --config {config 경로} --coco
```

## :two: Backbone
### :small_blue_diamond: 지원하는 모델
- [ ] Vgg16
- [ ] MobileNet-V1
- [ ] MobileNet-V2
- [ ] MobileNet-V3

### :small_blue_diamond: 지원하는 Dataset
- [ ] ImageNet
- [ ] OpenImagesV7
- [ ] PASS

### :small_blue_diamond: 성능
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
### :small_blue_diamond: 지원하는 모델
#### :radio_button: 2016
- [ ] YoloV1 (You Only Look Once, CVPR 2016)
- [x] SSD (Single Shot MultiBox Detector, ECCV 2016)
  - vgg16, mobilenet-v1, mobilenet-v2, mobilenet-v3
- [ ] YOLO9000 (YOLO9000: Better, Faster, Stronger, CVPR 2017)

#### :radio_button: 2019
- [ ] CenterNet (Objects as Points, CVPR 2019)


### :small_blue_diamond: 지원하는 Dataset
- [x] VOC2007+2012 (PASCAL VOC, 20 classes)
- [x] COCO2017 (Common Objects in Context, 80 classes)
- [x] Crowd Human (Crowd Human, 2 class)
- [x] Argoseye (Argoseye, 1 class)

### :small_blue_diamond: 성능
* 일반적으로 mAP는 다음과 같이 적용된다.
 - AP<sup>small</sup> : 32x32 이하의 작은 객체, IOU 0.5:0.95
 - AP<sup>medium</sup>: 32x32 이상, 96x96 이하의 객체, IOU 0.5:0.95
 - AP<sup>large</sup> : 96x96 이상의 큰 객체, IOU 0.5:0.95
* 이미지 입력 크기가 다양한 상황에서 위 평가지표는 적절하지 않으므로, 다음과 같이 적용된다.
 - AP<sup>small</sup> : bounding box의 넓이가 (1/6)<sup>2</sup> 이하인 객체
 - AP<sup>medium</sup>: bounding box의 넓이가 (1/6)<sup>2</sup> 이상, (1/3)<sup>2</sup> 이하인 객체
 - AP<sup>large</sup> : bounding box의 넓이가 (1/3)<sup>2</sup> 이상인 객체
* 그리고 IoU threshold는 다음과 같이 적용된다.
 - AP<sup>small</sup> : 0.5
 - AP<sup>medium</sup>: 0.6
 - AP<sup>large</sup> : 0.7
* 평가를 위해 11 point, 101 point, all point 계산 방법을 지원하지만, COCO에서 사용한 101 point 계산 방법을 사용한다.
* VOC 데이터셋의 경우 Occlusion 등의 이유로 Difficulty한 객체는 mAP 계산에서 제외하지만, 포함하여 계산되었다.

#### :radio_button: COCO2017 Dataset
|Model|Backbone|Params|Flops|pretrained|AP<sup>0.5:0.95</sup>|AP<sup>50</sup>|AP<sup>75</sup>|AP<sup>small</sup>|AP<sup>midium</sup>|AP<sup>large</sup>|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|-|-|-|-|
|SSD|VGG16|42.7M|37.5G|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V1|-|-|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V2|16.69M|2.25G|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V3|-|-|Imagenet|-|-|-|-|-|-|

#### :radio_button: VOC2007+2012 Dataset
|Model|Backbone|Params|Flops|pretrained|AP<sup>0.5:0.95</sup>|AP<sup>50</sup>|AP<sup>75</sup>|AP<sup>small</sup>|AP<sup>midium</sup>|AP<sup>large</sup>|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|-|-|-|-|
|SSD|VGG16|42.7M|37.5G|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V1|-|-|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V2|16.69M|2.25G|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V3|-|-|Imagenet|-|-|-|-|-|-|

#### :radio_button: CrowdHuman Dataset
|Model|Backbone|Params|Flops|pretrained|AP<sup>0.5:0.95</sup>|AP<sup>50</sup>|AP<sup>75</sup>|AP<sup>small</sup>|AP<sup>midium</sup>|AP<sup>large</sup>|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|-|-|-|-|
|SSD|VGG16|42.7M|37.5G|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V1|-|-|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V2|16.69M|2.25G|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V3|-|-|Imagenet|-|-|-|-|-|-|

#### :radio_button: Argoseye Dataset
|Model|Backbone|Params|Flops|pretrained|AP<sup>0.5:0.95</sup>|AP<sup>50</sup>|AP<sup>75</sup>|AP<sup>small</sup>|AP<sup>midium</sup>|AP<sup>large</sup>|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|YoloV1|-|-|-|-|-|-|-|-|-|-|
|SSD|VGG16|42.7M|37.5G|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V1|-|-|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V2|16.69M|2.25G|Imagenet|-|-|-|-|-|-|
|SSD|Mobilenet-V3|-|-|Imagenet|-|-|-|-|-|-|