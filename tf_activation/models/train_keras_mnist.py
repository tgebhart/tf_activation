import numpy as np
np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
import keras

import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import input_data


def reshape_dataset(data):
    ret = np.zeros(shape=(data.shape[0], 28, 28, 1))
    for i in range(data.shape[0]):
        ret[i, :, :, :] = np.reshape(data[i], [28,28,-1])
    return ret

def train(file_name, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    data = reshape_dataset(mnist.train.images)

    model = Sequential()
    winit = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=1)
    binit = keras.initializers.Constant(value=0.1)

    model.add(Conv2D(32, (5, 5), input_shape=data.shape[1:], padding='same', kernel_initializer=winit, bias_initializer=binit))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=winit, bias_initializer=binit))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=1))
    model.add(Dense(10, kernel_initializer=winit, bias_initializer=binit))

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=fn,
                  optimizer=adam,
                  metrics=['accuracy'])

    vals = reshape_dataset(mnist.test.images)

    model.fit(data, mnist.train.labels,
              batch_size=batch_size,
              validation_data=(vals, mnist.test.labels),
              nb_epoch=num_epochs,
              shuffle=False)

    if file_name != None:
        model.save(file_name)

    return model


if __name__ == "__main__":
    train("../../logdir/models/keras_mnist_cff", num_epochs=50)
