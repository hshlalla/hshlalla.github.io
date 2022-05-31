---
categories : 
- Tensorflow
title : "Image to npy"
tags:
- Numpy
- Data analysis
- Image preprocessing
last_modified_at:
---


# Image to Numpy
- 가끔 남의 코드를 보다보면 Numpy의 숨을 코드들을 잘사용하는경우가 많다. 틈나면 numpy는 자주자주 공부하자.
- image를 사용하다보면 큰양의 데이터를 사용할 경우 메모리 부족이 나나타 oom에러가 나는경우가 많다. 이를위해 데이터를 numpy를 이용저장해두고 노드해서 사용하면 메모리 사용량을 줄일 수 있다.

## import library


```python
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
```

- 빈 리스트와 딕셔너리를 11개 만들어 준다. 멀티인풋에 사용할 데이터를 만들기 위해 11개의 묶음을 만들자.


```python

labels = []
dataset_paths = {"1q_num" :{
    'AD': [],
    'MCI': []},
                 "2q_num" :{
    'AD': [],
    'MCI': []},
                 "3q_num" :{
    'AD': [],
    'MCI': []},
                 "4q_num" :{
    'AD': [],
    'MCI': []},
                 "5q_num" :{
    'AD': [],
    'MCI': []},
                 "6q_num" :{
    'AD': [],
    'MCI': []},
                 "7q_num" :{
    'AD': [],
    'MCI': []},
                 "8q_num" :{
    'AD': [],
    'MCI': []},
                 "9q_num" :{
    'AD': [],
    'MCI': []},
                 "10q_num" :{
    'AD': [],
    'MCI': []},
                 "11q_num" :{
    'AD': [],
    'MCI': []},
                 
    }
```

## Data Load
- Data load
- 1번 부터 11번까지의 이미지를 로드할예정.


```python
dataset_dir = 'Z:/dataset20220421_ver2/MCI_AD/'
dataset_list_dir = os.listdir(dataset_dir)
print(dataset_list_dir) 
```

    ['1', '10', '11', '2', '3', '4', '5', '6', '7', '8', '9']
    

## Data 생성
- for가 11번 시행됩니다.
- q_num_list_dir에는 2개의 라벨값이 들어가 있다.
- dataset_paths에 라벨 변수를 넣고


```python
for q_num in sorted(dataset_list_dir): 
    q_num_list_dir = os.listdir(os.path.join(dataset_dir, q_num))
    q_num_list_dir.pop(-2)
    #리스트에 불필요한 파일이 있어 pop을 이용해 지워줍니다.
    for label in q_num_list_dir:
        label_list_dir = os.listdir(os.path.join(dataset_dir, q_num, label))
        #dataset_paths의 딕셔너리에 label에 저장되어 있는 값[AD, MCI]을 넣고 그 키에다가 이미지 패스를 저장하는 코드 
        dataset_paths[str(q_num)+"q_num"][label].append([os.path.join(dataset_dir, q_num, label, file) for file in sorted(label_list_dir)])
```

- dataset_paths에는 11문제의 딕셔너리가 들어있습니다.
- 그 안에는 2가지 유형이 들어 있습니다.


```python
dataset_paths.keys()
```




    dict_keys(['1q_num', '2q_num', '3q_num', '4q_num', '5q_num', '6q_num', '7q_num', '8q_num', '9q_num', '10q_num', '11q_num'])




```python
dataset_paths["1q_num"].keys()
```




    dict_keys(['AD', 'MCI'])




```python
train_len = 160
for q_key in dataset_paths:
    train_dataset_x = None
    train_dataset_y = []
    valid_dataset_x = None
    valid_dataset_y = []
    for ix, key in enumerate(dataset_paths[q_key]):
        dataset_paths[q_key][key] = list(zip(*dataset_paths[q_key][key]))

        dataset_paths[q_key][key] = np.array([[tf.keras.preprocessing.image.img_to_array(image.load_img(q_num_img, target_size=(300, 300))).astype('float32')/255. for q_num_img in patient][0] for patient in dataset_paths[q_key][key]])
    #        dataset_paths[key] = np.array([patient for patient in zip(*dataset_paths[key])])
    #        print(dataset_paths[key].shape)
        #np.random.shuffle(dataset_paths[q_key][key])
        labels.append(key)
        train_dataset_x = dataset_paths[q_key][key][:train_len] if train_dataset_x is None else np.vstack((train_dataset_x, dataset_paths[q_key][key][:train_len]))
        valid_dataset_x = dataset_paths[q_key][key][train_len:] if valid_dataset_x is None else np.vstack((valid_dataset_x, dataset_paths[q_key][key][train_len:]))
        train_dataset_y += [ix]*train_len
        valid_dataset_y += [ix]*(len(dataset_paths[q_key][key]) - train_len)
        np.save("train_dataset_x" +str(q_key)+ ".npy", train_dataset_x)
        np.save('valid_dataset_x' +str(q_key)+ '.npy', valid_dataset_x)
        np.save('train_dataset_y' +str(q_key)+ '.npy', train_dataset_y)
        np.save('valid_dataset_y' +str(q_key)+ '.npy', valid_dataset_y)
```
