
# coding: utf-8

# # Recurrent Neural Networks

# In[22]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[23]:


from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM


# ## Exercise 2
# 
# RNN models can be applied to images too. In general we can apply them to any data where there's a connnection between nearby units. Let's see how we can easily build a model that works with images.
# 
# - Load the MNIST data, by now you should be able to do it blindfolded :)
# - reshape it so that an image looks like a long sequence of pixels
# - create a recurrent model and train it on the training data
# - how does it perform compared to a fully connected? How does it compare to Convolutional Neural Networks?
# 
# (feel free to run this exercise on a cloud GPU if it's too slow on your laptop)

# In[24]:


from keras.datasets import mnist


# In[25]:


(X_train, y_train), (X_test, y_test) = mnist.load_data('/tmp/mnist.npz')


# In[26]:


from keras.utils.np_utils import to_categorical


# In[27]:


X_train_r = X_train.reshape(-1,784)
X_test_r = X_test.reshape(-1,784)

X_train_r = X_train_r.astype('float')
X_test_r = X_test_r.astype('float')

X_train_r /= 255.0
X_test_r /= 255.0

X_test_r = X_test_r[:,None]
X_train_r = X_train_r[:,None]

y_test_cat = to_categorical(y_test)
y_train_cat = to_categorical(y_train)


# In[28]:


K.clear_session()

model = Sequential()

model.add(LSTM(6, input_shape=(1,784)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')


# In[29]:


model.summary()


# In[30]:


model.fit(X_train_r, y_train_cat, epochs=1, batch_size=1)


# In[31]:


y_pred = model.predict(X_test_r)


# In[32]:


model.evaluate(X_test_r, y_test_cat)

