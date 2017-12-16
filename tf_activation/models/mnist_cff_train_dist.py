from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import pickle

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
np.random.seed(2)
import tensorflow as tf
import matplotlib.pyplot as plt


from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer

persistence_module = tf.load_op_library('/home/tgebhart/python/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_train_persistence.so')

FLAGS = None
SAVE_PATH = '../../logdir/models'
p = 99
H = 0
EPOCHS = 50
DIAG_DIR = '../../logdir/data/experiments/reseeded_mnist_cff50_' + str(EPOCHS) + '_' + str(p)
if not os.path.exists(DIAG_DIR):
    os.makedirs(DIAG_DIR)

def plot_diagram(diag, n, i):

    ax = plt.subplot()

    ax.scatter(diag[:,0], diag[:,1], s=25, c=(diag[:,0] - diag[:,1])**2, cmap=plt.cm.coolwarm, zorder=10)
    lims = [
        np.min([0]),  # min of both axes
        np.max([1]),  # max of both axes
    ]

    # now plot both limits against eachother

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('Birth Time')
    plt.ylabel('Death Time')

    plt.savefig(os.path.join(n, 'diagram_' + str(i) + '.svg'), dpi=1200,
                            format='svg', bbox_inches='tight')

    plt.close()
    plt.clf()
    plt.cla()

def run(return_after=None):

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allocator_type = 'BFC'
    config.log_device_placement = False

    total_iterations = 20000
    epoch_size = EPOCHS
    train_sample_num = 55000
    num_steps = total_iterations // epoch_size
    batch_size = 128

    if return_after is None:
        return_after = num_steps

    with tf.device('/gpu:0'):
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

        sess.run(tf.global_variables_initializer())

        for epoch in range(epoch_size):

            acc_idx = np.random.randint(mnist.train.images.shape[0]-1, size=(1000))

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

            ps = percentiles.eval(feed_dict={x: mnist.train.images[acc_idx[:1]], keep_prob:1.0})

            diagram_filename = os.path.join(DIAG_DIR, 'diagram_' + str(epoch) + '.csv')

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
                                                                np.stack((ps, ps)),
                                                                H,
                                                                diagram_filename
                                                                )
            r = result.eval(feed_dict={x: mnist.train.images[acc_idx[:1]], keep_prob:1.0})

            diag = np.genfromtxt(diagram_filename, delimiter=',')

            plot_diagram(diag, DIAG_DIR, epoch)

            train_accuracy = accuracy.eval(feed_dict={x: mnist.train.images[acc_idx], y_: mnist.train.labels[acc_idx], keep_prob: 1.0})
            print("Epoch: %d, training accuracy %g" % (epoch, train_accuracy))
            for i in range(train_sample_num//batch_size):
                batch = mnist.train.next_batch(batch_size, shuffle=False)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0}))

        print('NOT saving ...')
        # save_path = saver.save(sess, os.path.join(SAVE_PATH, 'mnist_cff'+str(epoch_size)+'.ckpt'))
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
   initial = tf.truncated_normal(shape, stddev=0.1, seed=1)
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
    run()
