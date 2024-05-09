# 2024 월간 데이콘 저해상도 조류 이미지 분류 AI 경진 대회

<https://dacon.io/competitions/official/236251/overview/description>

데이콘에서 진행하는 저해상도 조류 이미지 분류 AI 경진 대회에 참가하였습니다.

좋은 성적을 거두진 못하였지만 이때까지 공부하였던 것들을 실제로 적용해 보는 좋은 기회가 된 것 같았습니다.

## 1.데이터셋
데이터셋은 TRAIN용 저해상도(64*64),고해상도(256*256) 조류이미지 각각 15,834장과 TEST용 저해상도(64*64) 조류이미지 6,786장이 있으며 종류는 25가지 입니다.

![distribution of bird species](https://github.com/k99885/dacon2024_bird_lowres_image_classification/assets/157681578/c10055e1-df92-4bd1-a068-8a10847f4657)

## 2.폴더 구조
```
├── train
│   ├── TRAIN_00000.jpg
│   ├── TRAIN_00001.jpg
│   ├── ...
│   └── TRAIN_xxxxx.jpg
├── test
│   ├── TEST_00000.jpg
│   ├── TEST_00001.jpg
│   ├── ...
│   └── TEST_xxxxx.jpg
├── sample_submission.csv
├── train.csv
├── test.csv
└── 
```
이와 같이 train,test 폴더가 있고 csv문서가 있습니다. 

csv파일안에는 데이터들의 label과 그에 해당하는 데이터의 주소가 적혀져있습니다.
![CSV](https://github.com/k99885/dacon2024_bird_lowres_image_classification/assets/157681578/9ad91819-2a45-4a31-93ab-b3af228d3203)
