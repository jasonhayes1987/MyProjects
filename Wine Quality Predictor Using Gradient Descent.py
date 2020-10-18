
# coding: utf-8

# # Gradient Descent

# In[13]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[14]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale


# In[15]:


import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import seaborn as sns


# In[16]:


from keras.optimizers import SGD, Adam, Adagrad, RMSprop


# ### Exercise 1
# 
# You've just been hired at a wine company and they would like you to help them build a model that predicts the quality of their wine based on several measurements. They give you a dataset with wine
# 
# - Load the ../data/wines.csv into Pandas
# - Use the column called "Class" as target
# - Check how many classes are there in target, and if necessary use dummy columns for a multi-class classification
# - Use all the other columns as features, check their range and distribution (using seaborn pairplot)
# - Rescale all the features using either MinMaxScaler or StandardScaler
# - Build a deep model with at least 1 hidden layer to classify the data
# - Choose the cost function, what will you use? Mean Squared Error? Binary Cross-Entropy? Categorical Cross-Entropy?
# - Choose an optimizer
# - Choose a value for the learning rate, you may want to try with several values
# - Choose a batch size
# - Train your model on all the data using a `validation_split=0.2`. Can you converge to 100% validation accuracy?
# - What's the minumum number of epochs to converge?
# - Repeat the training several times to verify how stable your results are

# In[17]:


wines = pd.read_csv('../data/wines.csv')


# In[18]:


wines['Class'].nunique()


# In[19]:


wines['Class']


# In[20]:


from keras.utils import to_categorical


# In[21]:


X = wines.drop('Class',axis=1)
y = to_categorical(wines['Class'], dtype=int)
y = y[:,1:]


# In[23]:


X.columns


# In[24]:


sns.pairplot(wines,hue='Class')


# In[25]:


sns.pairplot(X)


# In[26]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[27]:


standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()


# In[28]:


wines_minmax = minmax_scaler.fit(X).transform(X)
wines_standard = standard_scaler.fit_transform(X)
wines_standard


# In[29]:


plt.plot(wines_standard)


# In[30]:


K.clear_session

model = Sequential()
model.add(Dense(6, input_shape=(13,), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[31]:


model.summary()


# In[54]:


model.fit(wines_standard, y, epochs=20, batch_size=16, validation_split=.2)


# In[55]:


predictions = model.predict(wines_standard)


# In[56]:


predictions


# In[57]:


from sklearn.metrics import classification_report


# In[58]:


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


# In[59]:


accuracy = categorical_accuracy(y,predictions)


# ### Exercise 2
# 
# Since this dataset has 13 features we can only visualize pairs of features like we did in the Paired plot. We could however exploit the fact that a neural network is a function to extract 2 high level features to represent our data.
# 
# - Build a deep fully connected network with the following structure:
#     - Layer 1: 8 nodes
#     - Layer 2: 5 nodes
#     - Layer 3: 2 nodes
#     - Output : 3 nodes
# - Choose activation functions, inizializations, optimizer and learning rate so that it converges to 100% accuracy within 20 epochs (not easy)
# - Remember to train the model on the scaled data
# - Define a Feature Funtion like we did above between the input of the 1st layer and the output of the 3rd layer
# - Calculate the features and plot them on a 2-dimensional scatter plot
# - Can we distinguish the 3 classes well?
# 

# In[67]:


K.clear_session

model = Sequential()

model.add(Dense(8, input_shape=(13,), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(RMSprop(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])


# In[68]:


model.fit(wines_standard,y, epochs=20)


# In[69]:


result = model.evaluate(wines_standard, y)


# In[70]:


result


# In[42]:


model.layers


# In[72]:


K.clear_session()

model = Sequential()

model.add(Dense(8, input_shape=(13,), activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(2, activation='tanh'))
model.add(Dense(3, activation='sigmoid'))
model.compile(RMSprop(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])

inp = model.layers[0].input
out = model.layers[2].output

features_function = K.function([inp], [out])

plt.figure(figsize=(20,20))

for i in range(1,21):
    plt.subplot(5, 4, i)
    h = model.fit(wines_standard, y, epochs=1)
    test_accuracy = model.evaluate(wines_standard, y)[1]
    features = features_function([wines_standard])[0]
    plt.scatter(features[:,0], features[:,1], c=y, cmap='coolwarm')
    plt.title('Epoch: {}, Test Acc: {:3.1f} %'.format(i, test_accuracy * 100))
    
plt.tight_layout()


# ### Exercise 3
# 
# Keras functional API. So far we've always used the Sequential model API in Keras. However, Keras also offers a Functional API, which is much more powerful. You can find its [documentation here](https://keras.io/getting-started/functional-api-guide/). Let's see how we can leverage it.
# 
# - define an input layer called `inputs`
# - define two hidden layers as before, one with 8 nodes, one with 5 nodes
# - define a `second_to_last` layer with 2 nodes
# - define an output layer with 3 nodes
# - create a model that connect input and output
# - train it and make sure that it converges
# - define a function between inputs and second_to_last layer
# - recalculate the features and plot them

# In[73]:


from keras.layers import Input
from keras.models import Model


# In[74]:


K.clear_session()

inputs = Input(shape=(13,))
x = Dense(8, activation='tanh')(inputs)
x = Dense(5, activation='tanh')(x)
second_to_last = Dense(2, activation='tanh')(x)

outputs = Dense(3, activation='softmax')(second_to_last)

model = Model(inputs = inputs, outputs = outputs)

model.compile(RMSprop(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])

features_function = K.function([inputs], [second_to_last])

plt.figure(figsize=(20,20))

for i in range(1,21):
    plt.subplot(5, 4, i)
    h = model.fit(wines_standard, y, epochs=1)
    test_accuracy = model.evaluate(wines_standard, y)[1]
    features = features_function([wines_standard])[0]
    plt.scatter(features[:,0], features[:,1], c=y, cmap='coolwarm')
    plt.title('Epoch: {}, Test Acc: {:3.1f} %'.format(i, test_accuracy * 100))
    
plt.tight_layout()


# In[75]:


features_function = K.function([inputs], [second_to_last])


# In[76]:


features = features_function([wines_standard])[0]


# In[77]:


plt.scatter(x=features[:,0], y=features[:,1], c=y)


# ### Exercise 4 
# 
# Keras offers the possibility to call a function at each epoch. These are Callbacks, and their [documentation is here](https://keras.io/callbacks/). Callbacks allow us to add some neat functionality. In this exercise we'll explore a few of them.
# 
# - Split the data into train and test sets with a test_size = 0.3 and random_state=42
# - Reset and recompile your model
# - train the model on the train data using `validation_data=(X_test, y_test)`
# - Use the `EarlyStopping` callback to stop your training if the `val_loss` doesn't improve
# - Use the `ModelCheckpoint` callback to save the trained model to disk once training is finished
# - Use the `TensorBoard` callback to output your training information to a `/tmp/` subdirectory
# - Watch the next video for an overview of tensorboard

# In[78]:


X_train, X_test, y_train, y_test = train_test_split(wines_standard, y, test_size=0.3, random_state=42)


# In[79]:


from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


# In[80]:


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, mode='auto')
model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')
tensor_board = TensorBoard(log_dir='/tmp/udemy/tensorboard/')


# In[81]:


K.clear_session()

inputs = Input(shape=(13,))
x = Dense(8, activation='tanh')(inputs)
x = Dense(5, activation='tanh')(x)
second_to_last = Dense(2, activation='tanh')(x)

outputs = Dense(3, activation='softmax')(second_to_last)

model = Model(inputs = inputs, outputs = outputs)

model.compile(RMSprop(lr=.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=y_train, batch_size=16, epochs=20, callbacks=[early_stopping, model_checkpoint, tensor_board], validation_data=(X_test, y_test))


# In[82]:


model.summary()

