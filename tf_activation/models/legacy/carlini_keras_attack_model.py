import numpy as np
np.random.seed(1)
import tensorflow as tf
import os
import pickle
import gzip

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import keras


class MNISTModel:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        winit = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=1)
        binit = keras.initializers.Constant(value=0.1)

        model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), padding='same', kernel_initializer=winit, bias_initializer=binit))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(1024, kernel_initializer=winit, bias_initializer=binit))
        model.add(Activation('relu'))
        model.add(Dense(10, kernel_initializer=winit, bias_initializer=binit))
        model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)
