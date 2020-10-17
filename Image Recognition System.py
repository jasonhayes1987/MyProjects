
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


from keras.utils.np_utils import to_categorical


# In[3]:


from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K


# In[4]:


from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
from scipy import misc


# In[5]:


from keras.layers import Conv2D


# In[6]:


from keras.layers import MaxPool2D, AvgPool2D


# In[7]:


from keras.layers import Flatten, Activation


# ### Exercise 2
# 
# Pleased with your performance with the digits recognition task, your boss decides to challenge you with a harder task. Their online branch allows people to upload images to a website that generates and prints a postcard that is shipped to destination. Your boss would like to know what images people are loading on the site in order to provide targeted advertising on the same page, so he asks you to build an image recognition system capable of recognizing a few objects. Luckily for you, there's a dataset ready made with a collection of labeled images. This is the [Cifar 10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html), a very famous dataset that contains images for 10 different categories:
# 
# - airplane 										
# - automobile 										
# - bird 										
# - cat 										
# - deer 										
# - dog 										
# - frog 										
# - horse 										
# - ship 										
# - truck
# 
# In this exercise we will reach the limit of what you can achieve on your laptop and get ready for the next session on cloud GPUs.
# 
# Here's what you have to do:
# - load the cifar10 dataset using `keras.datasets.cifar10.load_data()`
# - display a few images, see how hard/easy it is for you to recognize an object with such low resolution
# - check the shape of X_train, does it need reshape?
# - check the scale of X_train, does it need rescaling?
# - check the shape of y_train, does it need reshape?
# - build a model with the following architecture, and choose the parameters and activation functions for each of the layers:
#     - conv2d
#     - conv2d
#     - maxpool
#     - conv2d
#     - conv2d
#     - maxpool
#     - flatten
#     - dense
#     - output
# - compile the model and check the number of parameters
# - attempt to train the model with the optimizer of your choice. How fast does training proceed?
# - If training is too slow (as expected) stop the execution and move to the next session!

# In[8]:


from keras.datasets import cifar10


# In[9]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[10]:


X_train = X_train.astype('float32') / 255.0


# In[11]:


X_test = X_test.astype('float32') / 255.0


# In[12]:


y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)


# In[13]:


K.clear_session()

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[14]:


model.summary()


# In[15]:


model.fit(X_train, y_train_cat, epochs=10, validation_split=.3, shuffle=True)


# In[16]:


model.evaluate(X_test, y_test_cat)

