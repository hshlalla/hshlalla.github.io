---
categories : 
- Tensorflow
title : "리눅스 서버에 venv 설치해보자"
tags:
- keras
- include_top
last_modified_at:
---

모델을 사용하다보면 인자값이 다양한데 모르고 지나가는 경우가 빈번하다. 처음 배울때는 그런가 보다 하고 넘어가는 것들이 누구를 알려줘야 하는 입장이 되면 하나하나 자세하게 알고 넘어가야 가르쳐 줄 때 오류가 없다.
모델을 하나 불러와서 include_top 어떻게 결과가 다른지 알아보자.


```python
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import VGG16
```

기본적으로 모델은 imagenet이라는 큰 데이터셋을 훈련셋으로 사용한다. imagenet은 1000개의 클래스로 나눠져 있어 모델훈련은 최종레이어에 classess가 1000개로 include를 True로 적용하게 되면 그 모델 그대로 적용된다. 때문에 동일한 데이터셋을 만들지 않는 이상 include를 True로 해둘 이유가 없다.
아래의 결과를 보자.


```python
input = Input(shape=(224, 224, 3))
model = VGG16(input_tensor=input, include_top=False, weights=None, pooling='max')
model.summary()
input = Input(shape=(224, 224, 3))
model = VGG16(input_tensor=input, include_top=True, weights=None, pooling='max')
model.summary()
```

    Model: "vgg16"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                     
     block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                     
     global_max_pooling2d (Globa  (None, 512)              0         
     lMaxPooling2D)                                                  
                                                                     
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    Model: "vgg16"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_3 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                     
     block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                     
     flatten (Flatten)           (None, 25088)             0         
                                                                     
     fc1 (Dense)                 (None, 4096)              102764544 
                                                                     
     fc2 (Dense)                 (None, 4096)              16781312  
                                                                     
     predictions (Dense)         (None, 1000)              4097000   
                                                                     
    =================================================================
    Total params: 138,357,544
    Trainable params: 138,357,544
    Non-trainable params: 0
    _________________________________________________________________
    

위에를 보면 2개가 차이가 나는것을 볼 수 있다.
그래서 대부분 include_top를 False로 해두고 model.output부분에 layer를 추가해서 사용한다.


```python
input = Input(shape=(224, 224, 3))
model = VGG16(input_tensor=input, include_top=False, weights=None, pooling='max')
 
x = model.output
x = Dense(1024, name='fully')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(512, )(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(3, activation='softmax', name='softmax')(x)
model = Model(model.input, x)
model.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_8 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                     
     block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                     
     global_max_pooling2d_3 (Glo  (None, 512)              0         
     balMaxPooling2D)                                                
                                                                     
     fully (Dense)               (None, 1024)              525312    
                                                                     
     batch_normalization_1 (Batc  (None, 1024)             4096      
     hNormalization)                                                 
                                                                     
     activation_1 (Activation)   (None, 1024)              0         
                                                                     
     dense (Dense)               (None, 512)               524800    
                                                                     
     batch_normalization_2 (Batc  (None, 512)              2048      
     hNormalization)                                                 
                                                                     
     activation_2 (Activation)   (None, 512)               0         
                                                                     
     softmax (Dense)             (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 15,772,483
    Trainable params: 15,769,411
    Non-trainable params: 3,072
    _________________________________________________________________
    

레이어를 위와같이 붙여서 사용한다.
transfer learning과 유사하게 보일 수도 있지만 pre training된 weight을 내려받어 통과시킨 feature을 그대로 사용하면서 학습시 fully connected계층만 트레이닝 시키는 transfer learning과는 명백히 다르다.
데이터가 imagenet과 유사할때는 transfer learning과 효과적이나 데이터가 완전히 다를때는 trainable를 True로해서 모든 레이어를 학습시켜야 한다.
하지만 이것도 데이터가 충분할때 가능한 이야기이고 데이터가 부족할 때는 trainable을 false로 놓는게 결과가 잘나온다.
