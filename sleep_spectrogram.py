# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:59:06 2021

@author: tamphi
"""
#%%
import pandas as pd
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
#%%
fname = "sleep_data.v64.npz"
with np.load(fname) as sdata:
    x_train = sdata['x_train']
    y_train = sdata['y_train']
    x_test  = sdata['x_test']
    y_test  = sdata['y_test']
    x_valid = sdata['x_valid']
    y_valid = sdata['y_valid']
#%%
# =============================================================================
# from sklearn.model_selection import train_test_split
# x_train, x_rem = train_test_split(x_train, test_size=0.2)
# y_train, y_rem = train_test_split(y_train, test_size=0.2)
# 
# x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.5)
# =============================================================================

#%%
def plot_digits(instances, images_per_row=1, **options):
    size = 64
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, **options)
    #plt.axis("off")
#%%
# plt.figure(figsize=(9,9))
# example_images = x_train[:10]
# plot_digits(example_images, images_per_row=5)
# plt.show()
# #%%
# i=68
# some_digit = x_train[i]
# some_digit_image = some_digit
# plt.imshow(some_digit_image)
# stage = str(y_train[i])
# plt.title("Stage: " + stage)
# plt.show()
#%%
#plot performance of ML model
def plot_model(history):
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    #summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#%%
#Make data 3 dimension
#pase info across 3 channels
X_train = np.stack([x_train, x_train, x_train],axis = -1)
X_valid = np.stack([x_valid,x_valid,x_valid],axis=-1)
X_test = np.stack([x_test,x_test,x_test],axis=-1)
#%%
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions 
#%%
# #RESNET50
# model = ResNet50(include_top=True, pooling='max', weights=None,input_shape = (64,64,3),classes = 5)
# model.summary()
# #%%
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history_1 = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
# #%%
# plot_model(history_1)
#%%
#DenseNet121
import tensorflow as tf
model_2 = tf.keras.applications.densenet.DenseNet121(include_top=True, pooling='max', weights=None,input_shape = (64,64,3),classes = 5)
model_2.summary()
#%%
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_2 = model_2.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

#%%
plot_model(history_2)