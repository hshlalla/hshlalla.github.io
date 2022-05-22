---
title : "Keras callback 함수를 알아보자"
categories : 
- Tensorflow
tags:
- Keras
- Data analysis
last_modified_at:
layout: archive
classes: layout--home
author_profile: false
---



# Keras Callbacks

keras로 학습을 시키다보면 다양한 옵션을 training 단계에서 줄 수 있다. 이제 수행하는 object를 callback라고 부른다. callback들을 통해서 tensorboard에 모든 batch of training들에 대해 metric 수치를 모니터링할 수도 있고, 이를 저장하는 것도 가능하다. Early Stop이나 Learning Rate Scheduling과 같은 기능을 통해 학습결과에 따라 학습을 멈추거나 학습률을 조정할수도 있다. 이처럼 Callback들을 잘 활용한다면, 딥러닝 학습의 결과를 보다 좋게 만들 수 있기 때문에, 많이 사용되는 callback 4가지를 소개하고, 사용법에 대해 포스팅하였다.

Keras 공식 설명 페이지 : [keras official documentation Callbakcs API](https://keras.io/api/callbacks/)


```python
import tensorflow as tf
import numpy as np
```

## LearningRateScheduler
tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
   LearningRateScheduler는 epoch에 따라 학습률을 조정하는 callback이다. 인자로 받는 schedule은 epoch index를 조정할 수 있는 function을 의미한다. 사용 예시는 다음과 같다.


```python
def scheduler(epoch, lr):
   if epoch < 10:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),epochs=15, callbacks=[callback], verbose=0)
```

   위의 예제에서는 epoch가 10보다 작을 경우에는 입력받은 learning rate를 그대로 사용하고, 10을 넘어가게 되면 lr을 줄여주는 연산을 수행한다. 일반적으로 학습이 진행되며 lr을 줄여나가기 때문에, 이런 방법을 사용하는것도 괜찮지만, 학습의 결과를 반영할 수 없고, 학습 시작전에 정한 값을 통해 lr을 줄여야한다는 단점이 있다. 다음에 소개할 callback은 이러한 단점이 보완되었다.

## ReduceLROnPlateau

Plateau는 안정기를 뜻한다. ReduceLROnPlateau는 말 그대로 학습이 진행되지 않는 안정기에 들어서면, learning rate에 변화를 준다. 동작하는 방법은 간단하다. 관찰 대상으로 삼은 metric이 정해진 epoch동안 일정 크기 이상 변화하지 않으면 lr을 정해진대로 변경한다. 파라미터의 monitor는 기준이 될 metrics를 의미하고,factor는 lr에 변화를 어느 정도 줄 것인지,patience는 몇 epoch동안 변화가 없으면 lr을 변화시킬건지, mode는 auto, min, max 3가지가 존재하는데, monitor로 정한 metric이 감소하는 방향으로 움직여야하는지, 증가하는 방향으로 움직여야하는지 정할 수 있다.

사용 예시는 다음 코드와 같다.


```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

# ModelCheckPoint

분석을 하다보면 verbose를 1로 놓고 한없이 쳐다 보는 일이 많다. 분석은 시간과의 싸움도 중요하다. 분석은 추세를 빠르게 판단하고 파라미터값을 수정하면서 시간을 절약하기위해 체크포인트는 꼭 사용하는것이 좋다.
ModelCheckPoint는 어떤 시점에서 model이나 weights를 저장할 것인지 정할 수 있는 callback이다. monitor로 넣은 metrics의 변화에 따라, best값을 저장하거나, model을 통째로 저장하거나, weights만을 저장하는 것도 가능하다. 사용 예시는 다음과 같다.


```python
model.compile(loss="mse", optimizer="Adam",
              metrics=['accuracy'])

EPOCHS = 10
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# 지금까지의 학습에서 가장 높은 acc를 보인 모델이 저장된다.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
```

# EarlyStopping

학습이 이루어지면 안정기에 들어가게 되고 train데이터에 더 fit하게 학습되어 validation 데이터가 loss가 오르는 순간이 발생한다. early stopping는 그 부분이 도달하게 되면 학습을 멈추어 최적화된 학습을 할 수 있도록 도와주는 callback 함수이다.


```python
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# 이 callback은 3개의 연속되는 epoch에서 validation loss의 변화가 없을 때  
# 학습을 중단합니다.
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=10, batch_size=1, callbacks=[callback],
                    verbose=0)
```
