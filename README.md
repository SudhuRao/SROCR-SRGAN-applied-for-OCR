# SROCR evaluation


```python
#Configure to run in GPU
import os
from tensorflow.python.client import device_lib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print(device_lib.list_local_devices())

```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 11452917310048345513
    ]
    

Import libraries and functions required.


```python
from glob import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

```


```python
from data_loader import alphabet as alphabet
from crnn_model import CRNN
from data_loader import load_data
```

    Using TensorFlow backend.
    


```python
from SROCR import build_generator as srocr_gen
```


```python
#CRNN configured to have input image as 256*32
input_width = 256
#input_width = 1024
input_height = 32
input_shape = (input_width, input_height, 1)

```


```python
def decode(chars):
    blank_char = '_'
    new = ''
    last = blank_char
    for c in chars:
        if (last == blank_char or last != c) and c != blank_char:
            new += c
        last = c
    return new
```

Import SRGAN generator trained for super-resolution for different settings.


```python
SROCR_gen_1x = srocr_gen([None,None, 1],2)
SROCR_gen_1x.load_weights('srocr_1xgen\generator_srocr.h5')
```

    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:174: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:181: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:186: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:190: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:199: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:206: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:1834: The name tf.nn.fused_batch_norm is deprecated. Please use tf.compat.v1.nn.fused_batch_norm instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:133: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.
    
    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:2018: The name tf.image.resize_nearest_neighbor is deprecated. Please use tf.compat.v1.image.resize_nearest_neighbor instead.
    
    


```python
srocr_crnn_1x = CRNN(input_shape, len(alphabet))
srocr_crnn_1x.load_weights('srocr_1xgen\crnn_srocr.h5')
```

    WARNING:tensorflow:From C:\Users\surao\AppData\Local\Continuum\anaconda3\envs\sudu_env\lib\site-packages\keras\backend\tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    
    


```python
SROCR_gen_4x = srocr_gen([None,None, 1],2)
SROCR_gen_4x.load_weights('srocr_4xgen\generator_srocr.h5')
```


```python
srocr_crnn_4x = CRNN(input_shape, len(alphabet))
srocr_crnn_4x.load_weights('srocr_4xgen\crnn_srocr.h5')
```


```python
SROCR_gen_10x = srocr_gen([None,None, 1],2)
SROCR_gen_10x.load_weights('srocr_10xgen\generator_srocr.h5')
```


```python
srocr_crnn_10x = CRNN(input_shape, len(alphabet))
srocr_crnn_10x.load_weights('srocr_10xgen\crnn_srocr.h5')
```


```python
#Import test images and ground truth

imgdir = 'C:/Users/surao/Desktop/Sudhu/Dataset/LHT_AERO_ALZEY_DATA/test_images'
gtdir = 'C:/Users/surao/Desktop/Sudhu/Dataset/LHT_AERO_ALZEY_DATA/test_gt'

high_reso,low_reso,labels,input_length,label_length,source_str = load_data(imgdir,gtdir)
```

# SROCR from LR images

The following section shows result of the output of generator, when a low resolution input is provided.


```python
def get_random_crop(image, crop_height, crop_width):


    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, 100)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop,x,y
```


```python
def plot_gan_zoom(G,name):
    crop_height = 64
    crop_width = 24
    fig = plt.figure(figsize=(12, 12))
    fig.set_tight_layout({"pad": .0})

    idx = np.random.randint(0, low_reso.shape[0] - 1)
    img_tmp = cv2.resize(low_reso[idx], (32, 256))
    img_hrp,x,y = get_random_crop(img_tmp, crop_height, crop_width)

    

    rect = patches.Rectangle((x,y),crop_height,crop_width,linewidth=1,edgecolor='r',facecolor='none')
    rect1 = patches.Rectangle((x,y),crop_height,crop_width,linewidth=1,edgecolor='r',facecolor='none')

    ax1  = fig.add_subplot(2, 2, 1)
    ax1.add_patch(rect)    
    #img_tmp = np.squeeze(img_tmp, axis=2)
    plt.imshow(img_tmp.transpose(1,0),cmap='gray')
    plt.grid('off')
    plt.axis('off')
    plt.title('bicubic')


    ax4  = fig.add_subplot(2, 2, 3)
    plt.imshow(img_hrp.transpose(1,0),cmap='gray')
    plt.grid('off')
    plt.axis('off')
    plt.title('bicubic')

    img = G.predict(np.expand_dims(low_reso[idx], axis=0) / 127.5 - 1)
    img_unnorm = (img + 1) * 127.5
    ax3  = fig.add_subplot(2, 2, 2)
    ax3.add_patch(rect1)
    img_gan = np.squeeze(img_unnorm, axis=0).astype(np.uint8)
    img_gan = np.squeeze(img_gan, axis=2)
    plt.imshow(img_gan.transpose(1,0),cmap='gray')

    #plt.imshow(img_gan)
    plt.grid('off')
    plt.axis('off')
    plt.title(name)


    ax6  = fig.add_subplot(2, 2, 4)
    plt.imshow(img_gan[y: y + crop_height, x: x + crop_width].transpose(1,0),cmap='gray')
    plt.grid('off')
    plt.axis('off')
    plt.title(name)
```


```python
def plot_srocr(G,ocr,name):
    n_imgs=4
    plt.figure(figsize=(12, 12))
    plt.tight_layout({"pad": .0})
    for i in range(0, n_imgs * 2, 2):
        idx = np.random.randint(0, low_reso.shape[0] - 1)
        plt.subplot(n_imgs, 2, i + 1)
        img_tmp = cv2.resize(low_reso[idx], (32, 256))
        plt.imshow(img_tmp.transpose(1,0),cmap='gray')
        plt.grid('off')
        plt.axis('off')
        plt.title('LR image ({})'.format(source_str[idx]))
    

        img = G.predict(np.expand_dims(low_reso[idx], axis=0) / 127.5 - 1)
        img_unnorm = (img + 1) * 127.5
        img_gan = np.squeeze(img_unnorm, axis=0).astype(np.uint8)
        img_gan = np.squeeze(img_gan, axis=2)
        plt.subplot(n_imgs, 2, i + 2)
        plt.imshow(img_gan.transpose(1,0),cmap='gray')  
        
        res = ocr.predict(img_unnorm)

        for i in range(len(res)):

            # best path, real ocr applications use beam search with dictionary and language model
            chars = [alphabet[c] for c in np.argmax(res[i], axis=1)]
            res_str = decode(chars)
    
        plt.grid('off')
        plt.axis('off')
        plt.title(name+ '({})'.format(res_str))
```

#### SROCR (learning rate from OCR = learning rate from SR) 


```python
plot_srocr(SROCR_gen_1x,srocr_crnn_1x,'SROCR ')
```


![png](output_22_0.png)



```python
plot_srocr(SROCR_gen_1x,srocr_crnn_1x,'SROCR ')
```


![png](output_23_0.png)



```python
plot_gan_zoom(SROCR_gen_1x,'SROCR')
```


![png](output_24_0.png)



```python
plot_gan_zoom(SROCR_gen_1x,'SROCR')
```


![png](output_25_0.png)


#### SROCR (learning rate from OCR = 4 x learning rate from SR)  


```python
plot_srocr(SROCR_gen_4x,srocr_crnn_4x,'SROCR ')
```


![png](output_27_0.png)



```python
plot_srocr(SROCR_gen_4x,srocr_crnn_4x,'SROCR ')
```


![png](output_28_0.png)



```python
plot_gan_zoom(SROCR_gen_4x,'SROCR')
```


![png](output_29_0.png)



```python
plot_gan_zoom(SROCR_gen_4x,'SROCR')
```


![png](output_30_0.png)


#### SROCR (learning rate from OCR = 10 x learning rate from SR)  


```python
plot_srocr(SROCR_gen_10x,srocr_crnn_10x,'SROCR ')
```


![png](output_32_0.png)



```python
plot_srocr(SROCR_gen_10x,srocr_crnn_10x,'SROCR ')
```


![png](output_33_0.png)



```python
plot_gan_zoom(SROCR_gen_10x,'SROCR')
```


![png](output_34_0.png)



```python
plot_gan_zoom(SROCR_gen_10x,'SROCR')
```


![png](output_35_0.png)



```python
plot_gan_zoom(G_ocr,'SROCR')
```


![png](output_36_0.png)


# SROCR vs SRGAN

The following section shows comparison of output of SROCR generator and SRGAN generator.


```python
def plot_gan_comp(G1,G2,G1_name,G2_name):
    n_imgs=4
    plt.figure(figsize=(18, 6))
    plt.tight_layout()
    for i in range(0, n_imgs * 3, 3):
        idx = np.random.randint(0, low_reso.shape[0] - 1)
        plt.subplot(n_imgs, 3, i + 1)
        plt.imshow(cv2.resize(low_reso[idx], (32, 256)).transpose(1,0),cmap='gray')
        plt.grid('off')
        plt.axis('off')
        plt.title('X2 (bicubic)')


        img = G1.predict(np.expand_dims(low_reso[idx], axis=0) / 127.5 - 1)
        img_unnorm = (img + 1) * 127.5
        img_gan = np.squeeze(img_unnorm, axis=0).astype(np.uint8)
        img_gan = np.squeeze(img_gan, axis=2)
        
        plt.subplot(n_imgs, 3, i + 2)
        plt.imshow(img_gan.transpose(1,0),cmap='gray')       
    
        plt.grid('off')
        plt.axis('off')
        plt.title(G1_name)
        
        img = G2.predict(np.expand_dims(low_reso[idx], axis=0) / 127.5 - 1)
        img_unnorm = (img + 1) * 127.5
        img_gan = np.squeeze(img_unnorm, axis=0).astype(np.uint8)
        img_gan = np.squeeze(img_gan, axis=2)
        
        plt.subplot(n_imgs, 3, i + 3)
        plt.imshow(img_gan.transpose(1,0),cmap='gray')        
    
        plt.grid('off')
        plt.axis('off')
        plt.title(G2_name)
```


```python
G_srgan = srocr_gen([None,None, 1],2)
G_srgan.load_weights('generator_srgan.h5')

```

#### SROCR (learning rate from OCR = learning rate from SR)  


```python
plot_gan_comp(SROCR_gen_1x,G_srgan,'SROCR','SRGAN')
```


![png](output_42_0.png)



```python
plot_gan_comp(SROCR_gen_1x,G_srgan,'SROCR','SRGAN')
```


![png](output_43_0.png)


#### SROCR (learning rate from OCR = 4 x learning rate from SR)  


```python
plot_gan_comp(SROCR_gen_4x,G_srgan,'SROCR','SRGAN')
```


![png](output_45_0.png)



```python
plot_gan_comp(SROCR_gen_4x,G_srgan,'SROCR','SRGAN')
```


![png](output_46_0.png)


#### SROCR (learning rate from OCR = 10 x learning rate from SR)  


```python
plot_gan_comp(SROCR_gen_10x,G_srgan,'SROCR','SRGAN')
```


![png](output_48_0.png)



```python
plot_gan_comp(SROCR_gen_10x,G_srgan,'SROCR','SRGAN')
```


![png](output_49_0.png)


# SROCR vs OCR

The following section shows comparison of the CRNN of SROCR  and CRNN trained on same dataset.

#### CRNN trained on low res images 


```python
#os.chdir('./crnn_low_res')
```


```python
import editdistance
import numpy as np

```


```python
from crnn_lr_data_loader import load_data as crnn_lr_load_data
from crnn_lr_data_loader import alphabet as alphabet
from crnn_model import CRNN
```


```python
imgs,labels,input_length,label_length,source_str = crnn_lr_load_data(imgdir,gtdir)
img_shape = (256, 32, 1)
crnn_lr = CRNN(img_shape, len(alphabet)) 
crnn_lr.load_weights('crnn_weights_epoch_500.h5') 
```


```python
mean_ed = 0
mean_ed_norm = 0
mean_character_recogniton_rate = 0
sum_ed = 0
char_count = 0
correct_word_count = 0 

word_recognition_rate = 0
images_ = np.ones([1, 256, 32, 1])
```


```python
j = 0
for idx in range(len(imgs)):

    
    images_[0] = imgs[idx]
    res = crnn_lr.predict(images_)

    for i in range(len(res)):
        j+=1
       
        # best path, real ocr applications use beam search with dictionary and language model
        chars = [alphabet[c] for c in np.argmax(res[i], axis=1)]
        gt_str = source_str[idx]
        res_str = decode(chars)
        
        ed = editdistance.eval(gt_str, res_str)
        #ed = levenshtein(gt_str, res_str)
        ed_norm = ed / len(gt_str)
        mean_ed += ed
        mean_ed_norm += ed_norm
        
        sum_ed += ed
        char_count += len(gt_str)
        if ed == 0.: correct_word_count += 1
        
        #print('%20s %20s %f' %(gt_str, res_str, ed))

mean_ed /= j
mean_ed_norm /= j
character_recogniton_rate = (char_count-sum_ed) / char_count
word_recognition_rate = correct_word_count / j
```


```python
print()
print('mean editdistance             %0.3f' % (mean_ed))
print('mean normalized editdistance  %0.3f' % (mean_ed_norm))
print('character recogniton rate     %0.3f' % (character_recogniton_rate))
print('word recognition rate         %0.3f' % (word_recognition_rate))
```

    
    mean editdistance             1.963
    mean normalized editdistance  0.291
    character recogniton rate     0.661
    word recognition rate         0.467
    

#### CRNN trained on high res images 


```python
os.chdir('../crnn_high_res')
```


```python
from crnn_hr_data_loader import load_data as crnn_hr_load_data
from crnn_hr_data_loader import alphabet as alphabet
from crnn_model import CRNN
```


```python
imgs,labels,input_length,label_length,source_str = crnn_hr_load_data(imgdir,gtdir)
img_shape = (256, 32, 1)
crnn_hr = CRNN(img_shape, len(alphabet)) 
crnn_hr.load_weights('crnn_weights_epoch_500.h5') 
```


```python
mean_ed = 0
mean_ed_norm = 0
mean_character_recogniton_rate = 0
sum_ed = 0
char_count = 0
correct_word_count = 0 

word_recognition_rate = 0
images_ = np.ones([1, 256, 32, 1])
```


```python
j = 0
for idx in range(len(imgs)):

    
    images_[0] = imgs[idx]
    res = crnn_hr.predict(images_)

    for i in range(len(res)):
        j+=1
       
        # best path, real ocr applications use beam search with dictionary and language model
        chars = [alphabet[c] for c in np.argmax(res[i], axis=1)]
        gt_str = source_str[idx]
        res_str = decode(chars)
        
        ed = editdistance.eval(gt_str, res_str)
        #ed = levenshtein(gt_str, res_str)
        ed_norm = ed / len(gt_str)
        mean_ed += ed
        mean_ed_norm += ed_norm
        
        sum_ed += ed
        char_count += len(gt_str)
        if ed == 0.: correct_word_count += 1
        
        #print('%20s %20s %f' %(gt_str, res_str, ed))

mean_ed /= j
mean_ed_norm /= j
character_recogniton_rate = (char_count-sum_ed) / char_count
word_recognition_rate = correct_word_count / j
```


```python
print()
print('mean editdistance             %0.3f' % (mean_ed))
print('mean normalized editdistance  %0.3f' % (mean_ed_norm))
print('character recogniton rate     %0.3f' % (character_recogniton_rate))
print('word recognition rate         %0.3f' % (word_recognition_rate))
```

    
    mean editdistance             1.735
    mean normalized editdistance  0.260
    character recogniton rate     0.700
    word recognition rate         0.515
    

#### SROCR text recognition perfromance (learning rate of OCR = learning rate of SR)


```python
os.chdir('..')
```


```python
from data_loader import load_data
from data_loader import alphabet as alphabet
from crnn_model import CRNN
```


```python
high_reso,low_reso,labels,input_length,label_length,source_str = load_data(imgdir,gtdir)

img_shape = (256, 32, 1)
srocr_1x_crnn = CRNN(img_shape, len(alphabet)) 
srocr_1x_crnn.load_weights('srocr_1xgen/crnn_srocr.h5') 
```


```python

mean_ed = 0
mean_ed_norm = 0
mean_character_recogniton_rate = 0
sum_ed = 0
char_count = 0
correct_word_count = 0 

word_recognition_rate = 0
images_ = np.ones([1, 256, 32, 1])
```


```python
j = 0
for idx in range(len(low_reso)):

    img_lr = low_reso[idx]
    img = SROCR_gen_1x.predict(np.expand_dims(img_lr, axis=0) / 127.5 - 1)
    img_unnorm = (img + 1) * 127.5
    res = srocr_1x_crnn.predict(img_unnorm)
    #images_[0] = low_reso[idx]
    #res = crnn.predict(images_)

    for i in range(len(res)):
        j+=1
       
        # best path, real ocr applications use beam search with dictionary and language model
        chars = [alphabet[c] for c in np.argmax(res[i], axis=1)]
        gt_str = source_str[idx]
        res_str = decode(chars)
        
        ed = editdistance.eval(gt_str, res_str)
        #ed = levenshtein(gt_str, res_str)
        ed_norm = ed / len(gt_str)
        mean_ed += ed
        mean_ed_norm += ed_norm
        
        sum_ed += ed
        char_count += len(gt_str)
        if ed == 0.: correct_word_count += 1
        
        #print('%20s %20s %f' %(gt_str, res_str, ed))

mean_ed /= j
mean_ed_norm /= j
character_recogniton_rate = (char_count-sum_ed) / char_count
word_recognition_rate = correct_word_count / j
```


```python
print()
print('mean editdistance             %0.3f' % (mean_ed))
print('mean normalized editdistance  %0.3f' % (mean_ed_norm))
print('character recogniton rate     %0.3f' % (character_recogniton_rate))
print('word recognition rate         %0.3f' % (word_recognition_rate))
```

    
    mean editdistance             1.927
    mean normalized editdistance  0.289
    character recogniton rate     0.667
    word recognition rate         0.456
    

#### SROCR text recognition perfromance (learning rate of OCR = 4 x learning rate of SR)


```python
img_shape = (256, 32, 1)
srocr_4x_crnn = CRNN(img_shape, len(alphabet)) 
srocr_4x_crnn.load_weights('srocr_4xgen/crnn_srocr.h5') 
```


```python

mean_ed = 0
mean_ed_norm = 0
mean_character_recogniton_rate = 0
sum_ed = 0
char_count = 0
correct_word_count = 0 

word_recognition_rate = 0
images_ = np.ones([1, 256, 32, 1])
```


```python
j = 0
for idx in range(len(low_reso)):

    img_lr = low_reso[idx]
    img = SROCR_gen_4x.predict(np.expand_dims(img_lr, axis=0) / 127.5 - 1)
    img_unnorm = (img + 1) * 127.5
    res = srocr_4x_crnn.predict(img_unnorm)
    #images_[0] = low_reso[idx]
    #res = crnn.predict(images_)

    for i in range(len(res)):
        j+=1
       
        # best path, real ocr applications use beam search with dictionary and language model
        chars = [alphabet[c] for c in np.argmax(res[i], axis=1)]
        gt_str = source_str[idx]
        res_str = decode(chars)
        
        ed = editdistance.eval(gt_str, res_str)
        #ed = levenshtein(gt_str, res_str)
        ed_norm = ed / len(gt_str)
        mean_ed += ed
        mean_ed_norm += ed_norm
        
        sum_ed += ed
        char_count += len(gt_str)
        if ed == 0.: correct_word_count += 1
        
        #print('%20s %20s %f' %(gt_str, res_str, ed))

mean_ed /= j
mean_ed_norm /= j
character_recogniton_rate = (char_count-sum_ed) / char_count
word_recognition_rate = correct_word_count / j
```


```python
print()
print('mean editdistance             %0.3f' % (mean_ed))
print('mean normalized editdistance  %0.3f' % (mean_ed_norm))
print('character recogniton rate     %0.3f' % (character_recogniton_rate))
print('word recognition rate         %0.3f' % (word_recognition_rate))
```

    
    mean editdistance             1.911
    mean normalized editdistance  0.287
    character recogniton rate     0.670
    word recognition rate         0.464
    

#### SROCR text recognition perfromance (learning rate of OCR = 10 x learning rate of SR)


```python
img_shape = (256, 32, 1)
srocr_10x_crnn = CRNN(img_shape, len(alphabet)) 
srocr_10x_crnn.load_weights('srocr_10xgen/crnn_srocr.h5') 
```


```python

mean_ed = 0
mean_ed_norm = 0
mean_character_recogniton_rate = 0
sum_ed = 0
char_count = 0
correct_word_count = 0 

word_recognition_rate = 0
images_ = np.ones([1, 256, 32, 1])
```


```python
j = 0
for idx in range(len(low_reso)):

    img_lr = low_reso[idx]
    img = SROCR_gen_10x.predict(np.expand_dims(img_lr, axis=0) / 127.5 - 1)
    img_unnorm = (img + 1) * 127.5
    res = srocr_10x_crnn.predict(img_unnorm)
    #images_[0] = low_reso[idx]
    #res = crnn.predict(images_)

    for i in range(len(res)):
        j+=1
       
        # best path, real ocr applications use beam search with dictionary and language model
        chars = [alphabet[c] for c in np.argmax(res[i], axis=1)]
        gt_str = source_str[idx]
        res_str = decode(chars)
        
        ed = editdistance.eval(gt_str, res_str)
        #ed = levenshtein(gt_str, res_str)
        ed_norm = ed / len(gt_str)
        mean_ed += ed
        mean_ed_norm += ed_norm
        
        sum_ed += ed
        char_count += len(gt_str)
        if ed == 0.: correct_word_count += 1
        
        #print('%20s %20s %f' %(gt_str, res_str, ed))

mean_ed /= j
mean_ed_norm /= j
character_recogniton_rate = (char_count-sum_ed) / char_count
word_recognition_rate = correct_word_count / j
```


```python
print()
print('mean editdistance             %0.3f' % (mean_ed))
print('mean normalized editdistance  %0.3f' % (mean_ed_norm))
print('character recogniton rate     %0.3f' % (character_recogniton_rate))
print('word recognition rate         %0.3f' % (word_recognition_rate))
```

    
    mean editdistance             2.675
    mean normalized editdistance  0.391
    character recogniton rate     0.538
    word recognition rate         0.402
    


```python

```
