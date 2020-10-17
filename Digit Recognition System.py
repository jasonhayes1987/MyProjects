
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[20]:


from keras.datasets import mnist


# In[21]:


from keras.utils.np_utils import to_categorical


# In[22]:


from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K


# In[23]:


from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
from scipy import misc


# In[24]:


from keras.layers import Conv2D


# In[25]:


from keras.layers import MaxPool2D, AvgPool2D


# In[26]:


from keras.layers import Flatten, Activation


# ### Exercise 1
# You've been hired by a shipping company to overhaul the way they route mail, parcels and packages. They want to build an image recognition system  capable of recognizing the digits in the zipcode on a package, so that it can be automatically routed to the correct location.
# You are tasked to build the digit recognition system. Luckily, you can rely on the MNIST dataset for the intial training of your model!
# 
# Build a deep convolutional neural network with at least two convolutional and two pooling layers before the fully connected layer.
# 
# - Start from the network we have just built
# - Insert a `Conv2D` layer after the first `MaxPool2D`, give it 64 filters.
# - Insert a `MaxPool2D` after that one
# - Insert an `Activation` layer
# - retrain the model
# - does performance improve?
# - how many parameters does this new model have? More or less than the previous model? Why?
# - how long did this second model take to train? Longer or shorter than the previous model? Why?
# - did it perform better or worse than the previous model?

# In[27]:


(X_train, y_train), (X_test, y_test) = mnist.load_data('/tmp/mnist.npz')


# In[28]:


X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)


# In[29]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0


# In[30]:


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[31]:


X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# In[32]:


model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[33]:


model.summary()


# In[34]:


model.fit(X_train, y_train_cat, batch_size=128, epochs=2, validation_split=.3)


# In[35]:


model.evaluate(X_test, y_test_cat)

