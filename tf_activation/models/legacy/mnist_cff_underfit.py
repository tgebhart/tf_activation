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
import pandas as pd

from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer

FLAGS = None
SAVE_PATH = '../../logdir/models'

def run(return_after=None):

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allocator_type = 'BFC'
    # config.log_device_placement = True

    total_iterations = 200
    epoch_size = 200
    num_steps = total_iterations // epoch_size
    batch_size = 200

    if return_after is None:
        return_after = num_steps

    with tf.device('/cpu:0'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        net, keep_prob = deepnn(x)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net['y_conv']))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(net['y_conv'], 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()

    with tf.Session() as sess:
        df = []
        num_epoch = 0
        sess.run(tf.global_variables_initializer())
        for i in range(total_iterations):
            batch = mnist.train.next_batch(batch_size, shuffle=False)
            if i % epoch_size == 0:
                col = {}
                train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                                        y_: batch[1], keep_prob: 1.0})
                train_ce = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

                col['accuracy'] = train_accuracy
                col['cross_entropy'] = train_ce
                col['epoch'] = i / epoch_size
                df.append(col)
                print('step %d, training accuracy %g, cross entropy %g' % (i, train_accuracy, train_ce))

                if num_epoch >= return_after:
                    break

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            num_epoch = num_epoch + 1

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0}))

        print('saving ...')
        df = pd.DataFrame(df)
        save_path = saver.save(sess, os.path.join(SAVE_PATH, 'mnist_cff_' + str(return_after) + '_underfit.ckpt'))
        df.to_pickle(os.path.join(SAVE_PATH, 'mnist_cff_' + str(return_after) + '_underfit.pkl'))
        print("model saved in file: {}".format(save_path))

def deepnn(x):
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
       ret['W_conv1'] = weight_variable([5, 5, 1, 32])
       ret['b_conv1'] = bias_variable([32])
       ret['h_conv1'] = tf.nn.relu(conv2d(ret['input'], ret['W_conv1']) + ret['b_conv1'])

       tf.summary.scalar('W_conv1', ret['W_conv1'])

       ret['h_conv1_reshaped'] = tf.reshape(ret['h_conv1'], [-1, 28*28*32])

       ret['W_fc1'] = weight_variable([28 * 28 * 32, 1024])
       ret['b_fc1'] = bias_variable([1024])

       ret['h_fc1'] = tf.nn.relu(tf.matmul(ret['h_conv1_reshaped'], ret['W_fc1']) + ret['b_fc1'])

       # Dropout - controls the complexity of the model, prevents co-adaptation of
       # features.
       keep_prob = tf.placeholder(tf.float32)
       ret['h_fc1_drop'] = tf.nn.dropout(ret['h_fc1'], keep_prob)

       # Map the 1024 features to 10 classes, one for each digit
       ret['W_fc2'] = weight_variable([1024, 10])
       ret['b_fc2'] = bias_variable([10])

       ret['y_conv'] = tf.matmul(ret['h_fc1_drop'], ret['W_fc2']) + ret['b_fc2']
       return ret, keep_prob


def conv2d(x, W):
   """conv2d returns a 2d convolution layer with full stride."""
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
   """max_pool_2x2 downsamples a feature map by 2X."""
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, name=None):
   """weight_variable generates a weight variable of a given shape."""
   initial = tf.truncated_normal(shape, stddev=0.1)
   if name is None:
       return tf.Variable(initial)
   return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
   """bias_variable generates a bias variable of a given shape."""
   initial = tf.constant(0.1, shape=shape)
   if name is None:
       return tf.Variable(initial)
   return tf.Variable(initial, name=name)


if __name__ == "__main__":
    try:
        return_after = int(sys.argv[1])
    except IndexError:
        return_after = None
    run(return_after=return_after)
