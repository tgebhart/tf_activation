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

SAVE_PATH = '../../logdir/models'
ADV_PATH = '../../logdir/adversaries'
ELITE_PATH = '../../logdir/elites'

persistence_module = tf.load_op_library('/home/tgebhart/Projects/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')

def run(return_after=None):

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allocator_type = 'BFC'
    config.log_device_placement = True

    # one_adversary = np.genfromtxt(os.path.join(ADV_PATH, 'mnist_1_10:32:44_20-07-17.csv'), delimiter=',')
    # one_adversary = np.genfromtxt(os.path.join(ADV_PATH, 'mnist_8_15:59:29_04-08-17.csv'), delimiter=',')
    # one_adversary = np.genfromtxt(os.path.join(ELITE_PATH, 'mnist_8_16:13:32_04-08-17.csv'), delimiter=',')

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

    with tf.Session(config=config) as sess:

        saver.restore(sess, os.path.join(SAVE_PATH, 'mnist_cff_2000.ckpt'))

        # test_inputs = np.stack((mnist.test.images[2],one_adversary))
        test_inputs = np.stack((mnist.test.images[2],mnist.test.images[61]))


        acc = accuracy.eval(feed_dict={
            x: test_inputs, y_: mnist.test.labels[2:3], keep_prob: 1.0})


        p = 99


        percentiles = persistence_module.layerwise_percentile([net['input'],
                                                        net['W_conv1'],
                                                        net['h_conv1'],
                                                        net['h_conv1'],
                                                        net['W_fc1'],
                                                        net['h_fc1'],
                                                        net['h_fc1_drop'],
                                                        net['W_fc2'],
                                                        net['y_conv']],
                                                        [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                        [p,p,p])

        ps1 = percentiles.eval(feed_dict={x: test_inputs[1:2], keep_prob:1.0});
        print(ps1)

        ps2 = percentiles.eval(feed_dict={x: test_inputs[1:2], keep_prob:1.0});
        print(ps2)

        result = persistence_module.input_graph_persistence([net['input'],
                                                            net['W_conv1'],
                                                            net['h_conv1'],
                                                            net['h_conv1'],
                                                            net['W_fc1'],
                                                            net['h_fc1'],
                                                            net['h_fc1_drop'],
                                                            net['W_fc2'],
                                                            net['y_conv']],
                                                            [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                            np.stack((ps1, ps2))
                                                            )
        r = result.eval(feed_dict={x: test_inputs[1:], keep_prob:1.0})
        print(r)

        #
        # print(np.stack((ps1, ps2)).shape)
        #

        # result = persistence_module.bottleneck_distance([net['input'],
        #                                                 net['W_conv1'],
        #                                                 net['h_conv1'],
        #                                                 net['h_conv1'],
        #                                                 net['W_fc1'],
        #                                                 net['h_fc1'],
        #                                                 net['h_fc1_drop'],
        #                                                 net['W_fc2'],
        #                                                 net['y_conv']],
        #                                                 [0, 1, 2, 2, 1, 4, 4, 1, 4],
        #                                                 np.stack((ps1, ps2))
        #                                                 )
        #
        #
        # r = result.eval(feed_dict={x: test_inputs, keep_prob:1.0})
        # print('distance:', r)

        print('Test accuracy: {}'.format(acc))
        print('for labels: ', mnist.test.labels[2], mnist.test.labels[5])

        # print('saving ...')
        # save_path = saver.save(sess, os.path.join(SAVE_PATH, 'mnist_cff_' + str(return_after) + '.ckpt'))
        # print("model saved in file: {}".format(save_path))

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
