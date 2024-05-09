# 2024 월간 데이콘 저해상도 조류 이미지 분류 AI 경진 대회

<https://dacon.io/competitions/official/236251/overview/description>

데이콘에서 진행하는 저해상도 조류 이미지 분류 AI 경진 대회에 참가하였습니다.

좋은 성적을 거두진 못하였지만 이때까지 공부하였던 것들을 실제로 적용해 보는 좋은 기회가 된 것 같았습니다.

## 1. 데이터셋
데이터셋은 TRAIN용 저해상도(64*64),고해상도(256*256) 조류이미지 각각 15,834장과 TEST용 저해상도(64*64) 조류이미지 6,786장이 있으며 종류는 25가지 입니다.

![sample](https://github.com/k99885/dacon2024_bird_lowres_image_classification/assets/157681578/caf51a63-11b6-40ba-9c82-aadb6b0d172f)

![distribution of bird species](https://github.com/k99885/dacon2024_bird_lowres_image_classification/assets/157681578/c10055e1-df92-4bd1-a068-8a10847f4657)

## 2. 폴더 구조
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

## 3. 데이터 로드
```
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=10,
    horizontal_flip=True
)
val_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    )
```
데이터 전처리와 데이터 증강을 위하여 ImageDataGenerator를 생성하였습니다.

```
TARGET_SIZE=(256,256)
BATCH_SIZE=64
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation="lanczos" # 보간 방법 설정

)

val_images = val_generator.flow_from_dataframe(
    dataframe=val_df,
    x_col='Filepath',
    y_col='label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation="lanczos" # 보간 방법 설정


)
```
생성된 ImageDataGenerator에서 flow_from_dataframe을 호출하여 데이터를 로드해준후에 여러가지 파라메타들을 설정해 주었습니다.

## 4. 모델 생성
여러가지 CNN 기반 모델들을 테스트 해본결과 EfficientNetV2B3 의 성능이 가장 좋았습니다.
```
pretrained_model = tf.keras.applications.EfficientNetV2B3 (
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet',
    pooling='max'
)

# Freezing the layers of a pretrained neural network
for i, layer in enumerate(pretrained_model.layers):
    pretrained_model.layers[i].trainable = False
```
모델 서브클래싱을 위해서 base모델을 EfficientNetV2B3으로 로드하고 모델의 가중치는 새로 학습하지않고 이미지넷의 학습된 가중치를 그대로 사용하였습니다.
```
inputs = layers.Input(shape = (256,256,3), name='inputLayer')
pretrain_out = pretrained_model(inputs, training = False)
initializer = tf.keras.initializers.HeNormal()  # He 초기화 객체 생성

x = layers.Dense(256,kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.0001))(pretrain_out)
x = layers.Activation(activation="relu")(x)
x = BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(25)(x)
outputs = layers.Activation(activation="softmax", dtype=tf.float32, name='activationLayer')(x)
model = Model(inputs=inputs, outputs=outputs)
```
base 모델에서 include_top=False 을 해주었으니 완전연결층부분을 모델링 해주었습니다.

또한 과적합 방지를 위하여 HeNormal()로 가중치를 초기화하고 BatchNormalization(),.Dropout(0.4)을 적용하여 모델 성능을 향상시켰습니다.

![model summary](https://github.com/k99885/dacon2024_bird_lowres_image_classification/assets/157681578/98bb5460-1588-49a7-876d-588b58a70d9a)

input의 256*256의 데이터가 25개의 종류로 분류되는 모델을 생성하고 훈련 가능한 가중치의 개수는 40만 개 정도인 것을 확인하였습니다.

```
model.compile(
    optimizer=Adam(0.1),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
생성한 model을 컴파일 시켜줍니다. optimizer는 adam을 사용하고 손실함수로 sparse_categorical_crossentropy을 사용하였습니다.

## 5. 모델훈련

```
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.1,
    verbose=1)

CALLBACKS = [early_stopping_callback, reduce_lr_callback]
```
우선 최적화를 위하여 callback함수를 설정해줍니다.
```
history = model.fit(
    train_images,
    steps_per_epoch=len(train_images),
    validation_data=val_images,
    validation_steps=len(val_images),
    epochs=10,
    callbacks=CALLBACKS
)
```
에포크를 10번씩 설정하고 여러번 훈련시켜주었습니다.

![FIT](https://github.com/k99885/dacon2024_bird_lowres_image_classification/assets/157681578/e42c1ed3-407c-4123-b87a-e33d47f18ad2)

약 98 정도의 accuracy와 96 의 val_accuracy를 얻었습니다.



