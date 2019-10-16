#!/usr/bin/env python
# coding: utf-8

# In[250]:


import tensorflow as tf
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# # Load features
# 

# In[256]:


# Load features

data_file = 'train2/labels'
features_dir = 'train2/features/'

ext = '.txt'

labels = []
features = []
file_names = []
files_path = []

print("Load file name and label from file:", data_file)

for path, subdirs, files in os.walk(data_file):   
    for name in files: 
        
        path_to_file = features_dir + os.path.splitext(name)[0]

        with open('train2/labels/{}.csv'.format(os.path.splitext(name)[0])) as f1 :
        
            reference = np.genfromtxt(f1, delimiter=' ', dtype='str')
            #print(reference)
            print("Load features and labels from file names: ", os.path.splitext(name)[0])
            for idx, name in enumerate(reference[:,0]):
                #print(name)
                file_name = name.split('.')[0]
                #print(file_name)
                                                                
                with open(path_to_file +'/{}.txt'.format(os.path.splitext(file_name)[0])) as f2:
                    #print(f2)
                    feature = np.loadtxt(f2)
                    labels.append(int(reference[idx,1]))
                    file_names.append(file_name)
                    features.append(feature)
                    
print('Features loaded')

print('length labels = ', len(labels))
print('length features = ', len(features))                
                
                



# In[257]:


#print(features[0])
print(features[0].shape)
print(len(file_names))


# In[258]:


# Shuffle data

p = np.random.permutation(len(labels))
print(p)
labels = [labels[j] for j in p]
file_names = [file_names[j] for j in p]
features = [features[j] for j in p]
#print (features)
#print(categories)
print(labels[0])
print(file_names[0])


# In[259]:


# Normalization

max_features=[]
for i in range(len(features)): 
    #print(np.max(features[i]))
    max_features.append(np.max(features[i]))
val_max = np.max(max_features)
print('la valeur maximale est: ',val_max)

for i in range(len(features)):
    for j in features[i]:
        j /= val_max

# print(features)


# In[260]:


# Make numpy array from lists

features = np.array(features)
labels = np.array(labels)


# In[262]:


import tensorflow.keras as keras
import tensorflow as tf
print(tf.__version__)
from keras.utils import to_categorical


# In[263]:


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[264]:


# Model 2
# Issue de : https://github.com/Shahnawax/AER-CNN-KERAS/blob/master/train_network.py

from keras.optimizers import SGD
#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

tf.keras.backend.clear_session()

inputShape = (128,431,1)

numClasses = 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3),input_shape=inputShape,activation='relu',padding='valid'))
# adding a batch normalization and maxpooling layer 3 by 5 (Note: We are compressing the 2nd diemension more in order to get a more square shape)
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,5)))
# adding second convolutionial layer with 32 filters and 1 by 3 kernal size, using the rectifier linear unit as the activation
model.add(tf.keras.layers.Conv2D(32, (1,3),activation='relu',padding = 'valid'))
# adding the first convolutionial layer with 32 filters and 3 by 3 kernal size, using the rectifier linear unit as the activation
model.add(tf.keras.layers.Conv2D(32, (3,3),activation='relu',padding = 'valid'))
# adding batch normalization layer
#model.add(tf.keras.layers.BatchNormalization())
# adding forth convolutionial layer with 32 filters and 1 by 3 kernal size, using the rectifier linear unit as the activation
model.add(tf.keras.layers.Conv2D(32, (1,3),activation='relu',padding = 'valid'))
# adding batch normalization and max pooling layers
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,3)))
# adding fift convolutionial layer with 64 filters and 3 by 3 kernal size, using the rectifier linear unit as the activation
model.add(tf.keras.layers.Conv2D(32, (3,3),activation='relu',padding = 'valid'))
# adding batch normalization and max pooling layers
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# flattening the output in order to apply the fully connected layer
model.add(tf.keras.layers.Flatten())
# adding a drop out for the regularizaing purposes
model.add(tf.keras.layers.Dropout(0.2))
#adding a fully connected layer 128 filters
model.add(tf.keras.layers.Dense(128, activation='relu'))   
# adding a drop out for the regularizaing purposes
model.add(tf.keras.layers.Dropout(0.2))
# adding softmax layer for the classification
model.add(tf.keras.layers.Dense(numClasses, activation='sigmoid'))
# Compiling the model to generate a model

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

print('Network loaded')


# In[265]:


#callback = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=10)] 
callback = [] 

validation_split = 0.2 # Fraction of the training data to be used as validation data


shuffle = True
batch_size=20



epochs=30


history= model.fit(x=features[:, :, :, np.newaxis], y=labels, epochs=epochs, batch_size=batch_size, callbacks=callback,                    validation_split=validation_split, shuffle=shuffle)


# In[266]:


# Evaluation

model.evaluate(features[:,:,:,np.newaxis], labels, batch_size=batch_size)


# In[ ]:




