from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from parse import *
import sys
import os
import pickle

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer

class Base_Model(object):

    def __init__(self, activation='relu',
                strides=[1, 1, 1, 1],
                seed = None,
                stddev = 0.1,
                initializer='normal'):

        self.activation = activation
        self.strides = strides
        self.stddev = stddev
        self.seed = seed
        self.initializer = initializer
        self.implements_dropout = False

        self.trained_model_string = '{}_{}{:.1e}_{}epochs_{}batch'

        if self.seed is not None:
            np.random.seed(self.seed)

        self.graph = {}

    def get_model_string(self):
        return 'base'

    def get_trained_model_string(self, dataset, optimizer, learning_rate, epochs, batch_size):
        return self.trained_model_string.format(dataset, optimizer, learning_rate,
                str(epochs), str(batch_size)).replace('.','p')+'.ckpt'

    def infer_training_parameters_from_filename(self, s):
        if s.rfind('/') != -1:
            s = s[:s.rfind('/')]
        # hack
        s = s.replace('p','.')
        s = s.replace('e.ochs', 'epochs')
        print(s)

        p = parse(self.trained_model_string, s)

        return {'dataset' : p[0],
                'optimizer' : p[1],
                'learning_rate' : p[2],
                'epochs' : p[3],
                'batch_size' : p[4]}

    def train_mnist(self, save_loc, total_iterations=20000, epoch_size=50, train_sample_num=55000,
                    batch_size=128, return_after=None, optimizer='SGD', learning_rate=1e-4,
                    test_on=1000):

        directory = os.path.join(save_loc, self.get_model_string())
        if not os.path.exists(directory):
            os.makedirs(directory)

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # config.gpu_options.allocator_type = 'BFC'
        config.log_device_placement = False

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        num_steps = total_iterations // epoch_size

        if return_after is None:
            return_after = num_steps

        with tf.device('/gpu:0'):
            # Create the model
            x = tf.placeholder(tf.float32, [None, 784])

            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 10])

            # Build the graph for the deep net
            net = self.build_graph(x)

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net['y']))
            train_step = self.get_train_step(cross_entropy, optimizer=optimizer, lr=learning_rate)
            correct_prediction = tf.equal(tf.argmax(net['y'], 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.device('/cpu:0'):

            saver = tf.train.Saver()

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            for epoch in range(epoch_size):
                acc_idx = np.random.randint(mnist.train.images.shape[0]-1, size=(1000))
                train_accuracy = accuracy.eval(feed_dict={x: mnist.train.images[acc_idx], y_: mnist.train.labels[acc_idx]})
                print("Epoch: %d, training accuracy %g" % (epoch, train_accuracy))
                for i in range(train_sample_num//batch_size):
                    batch = mnist.train.next_batch(batch_size, shuffle=False)
                    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: mnist.test.images[:test_on], y_: mnist.test.labels[:test_on]}))

            print('saving ...')
            save_name = self.get_trained_model_string('mnist', optimizer, learning_rate, epoch_size, batch_size)
            save_path = saver.save(sess, os.path.join(directory, save_name))
            print("model saved in file: {}".format(save_path))

    def activate(self, l):
        if self.activation == 'relu':
            return tf.nn.relu(l)
        if self.activation == 'sigmoid':
            return tf.nn.sigmoid(l)
        else:
            raise ValueError('please choose valid activation function')

    def conv2d(self, x, W):
       """conv2d returns a 2d convolution layer with full stride."""
       return tf.nn.conv2d(x, W, strides=self.strides, padding='SAME')

    def weight_variable(self, shape, name=None):
       """weight_variable generates a weight variable of a given shape."""
       if self.initializer == 'truncated_normal':
           initial = tf.truncated_normal(shape, stddev=self.stddev, seed=self.seed)
       if self.initializer == 'normal':
           initial = tf.random_normal(shape, stddev=self.stddev, seed=self.seed)
       else:
           initial = tf.random_uniform(shape, minval=-self.stddev, maxval=self.stddev, seed=self.seed)
       if name is None:
           return tf.Variable(initial)
       return tf.Variable(initial, name=name)

    def get_train_step(self, loss, optimizer='SGD', lr=1e-4):
        if optimizer == 'ADAM':
            return tf.train.AdamOptimizer(lr).minimize(loss)
        if optimizer == 'SGD':
            return tf.train.GradientDescentOptimizer(lr).minimize(loss)
        else:
            raise ValueError('please choose valid optimizer')



if __name__ == "__main__":
    print('Base Model')
