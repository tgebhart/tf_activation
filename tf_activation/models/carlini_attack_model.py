from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import pickle

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer

SAVE_PATH = '../logdir/models'

class MNISTModel:

    def __init__(self, session=None, total_iterations=20000, epoch_size=100, batch_size=100):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        with tf.device('/gpu:0'):

            self.x = tf.placeholder(tf.float32, [None, 784])

            # Define loss and optimizer
            self.y_ = tf.placeholder(tf.float32, [None, 10])

            # Build the graph for the deep net
            self.net, self.keep_prob = self.deepnn(self.x)

            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.net['y_conv']))
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
            self.correct_prediction = tf.equal(tf.argmax(self.net['y_conv'], 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.sess = session

        if self.sess is None:
            self.sess = tf.Session(config=self.config)

        self.total_iterations = total_iterations
        self.epoch_size = epoch_size
        self.num_steps = total_iterations // epoch_size
        self.batch_size = batch_size

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allocator_type = 'BFC'
        config.log_device_placement = False
        self.config = config


    def restore(self, restore):
        with tf.device('/cpu:0'):
            saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(SAVE_PATH, restore))

    def deepnn(self, x):
       """deepnn builds the graph for a deep net for classifying digits.

       Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.

       Returns:
           A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
           equal to the logits of classifying the digit into one of 10 classes (the
           digits 0-9). keep_prob is a scalar placeholder for the probability of
           dropout.
       """

       with tf.device('/cpu:0'):
           ret = {}
           # Reshape to use within a convolutional neural net.
           # Last dimension is for "features" - there is only one here, since images are
           # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
           ret['input'] = tf.reshape(x, [-1, 28, 28, 1], name='input')

           # First convolutional layer - maps one grayscale image to 32 feature maps.
           ret['W_conv1'] = self.weight_variable([5, 5, 1, 32])
           ret['b_conv1'] = self.bias_variable([32])
           ret['h_conv1'] = tf.nn.relu(self.conv2d(ret['input'], ret['W_conv1']) + ret['b_conv1'])

           ret['h_conv1_reshaped'] = tf.reshape(ret['h_conv1'], [-1, 28*28*32])

           ret['W_fc1'] = self.weight_variable([28 * 28 * 32, 1024])
           ret['b_fc1'] = self.bias_variable([1024])

           ret['h_fc1'] = tf.nn.relu(tf.matmul(ret['h_conv1_reshaped'], ret['W_fc1']) + ret['b_fc1'])

           # Dropout - controls the complexity of the model, prevents co-adaptation of
           # features.
           keep_prob = tf.placeholder(tf.float32)
           ret['h_fc1_drop'] = tf.nn.dropout(ret['h_fc1'], keep_prob)

           # Map the 1024 features to 10 classes, one for each digit
           ret['W_fc2'] = self.weight_variable([1024, 10])
           ret['b_fc2'] = self.bias_variable([10])

           ret['y_conv'] = tf.matmul(ret['h_fc1'], ret['W_fc2']) + ret['b_fc2']
           return ret, keep_prob

    def conv2d(self, x, W):
       """conv2d returns a 2d convolution layer with full stride."""
       return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
       """max_pool_2x2 downsamples a feature map by 2X."""
       return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape, name=None):
       """weight_variable generates a weight variable of a given shape."""
       initial = tf.truncated_normal(shape, stddev=0.1)
       if name is None:
           return tf.Variable(initial)
       return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name=None):
       """bias_variable generates a bias variable of a given shape."""
       initial = tf.constant(0.1, shape=shape)
       if name is None:
           return tf.Variable(initial)
       return tf.Variable(initial, name=name)

    def train(self):

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        with tf.device('/cpu:0'):

            saver = tf.train.Saver()

        num_epoch = 0
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.total_iterations):
            batch = mnist.train.next_batch(self.batch_size)
            if i % self.epoch_size == 0:
                train_accuracy = self.accuracy.eval(feed_dict={self.x: batch[0],
                                                        self.y_: batch[1]})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})
            num_epoch = num_epoch + 1

        print('test accuracy %g' % self.accuracy.eval(feed_dict={
            self.x: mnist.test.images[:1000], self.y_: mnist.test.labels[:1000]}))

        print('saving ...')
        save_path = saver.save(self.sess, os.path.join(SAVE_PATH, 'carlini_attack.ckpt'))
        print("model saved in file: {}".format(save_path))


    def predict(self, data):
        ndata = self.sess.run(data)
        ndata = np.reshape(ndata, [-1, 784])
        convs = self.net['y_conv'].eval(feed_dict={self.x: ndata, self.keep_prob:1.0})
        convs = tf.convert_to_tensor(convs, dtype=np.float32)
        return convs
