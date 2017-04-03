import os
import csv
import cv2
import numpy as np
import sklearn
import utils
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# My model is based on the CNN as described in the
# Nvidia End to End Learning for Self-Driving Cars paper
# Reference: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

# Five convolutional layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())

# Five fully connected layers
model.add(Dropout(0.5))
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.summary()

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
model.compile(loss='mse', optimizer=adam)

train_generator = utils.generate_batch()
validation_generator = utils.generate_batch()

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=20032,
                                     nb_epoch=10,
                                     validation_data=validation_generator,
                                     nb_val_samples=6400,
                                     verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
print('Model saved, completed running model.py')
