from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import listdir
from os.path import isfile
import pickle

import tensorflow as tf
import numpy as np

FLAGS = None
SAVE_PATH = '../../logdir/models'

model = 'mnist_cff_2000.ckpt'
SAVE_PATH = '../../logdir/models'
ELITE_LOC = '../../logdir/elites/mnist_test_elites_19:45:10_18-09-17'
DIAG_DIR = os.path.join(ELITE_LOC, 'diagrams')
TRUE_DIR = os.path.join(DIAG_DIR, 'true')
FALSE_DIR = os.path.join(DIAG_DIR, 'elites')

TAKE = 20

def get_batch(batch_size, i, arr):
    sidx = (i*batch_size) % arr.shape[0]
    eidx = (sidx + batch_size) % arr.shape[0]
    if eidx <= sidx:
        return np.append(arr[sidx:,:], arr[:eidx,:], axis=0)
    return arr[sidx:eidx,:]

def run(model_name):

    true_files = [f for f in listdir(TRUE_DIR) if isfile(os.path.join(TRUE_DIR, f))]
    false_files = [f for f in listdir(FALSE_DIR) if isfile(os.path.join(FALSE_DIR, f))]

    dummy = np.genfromtxt(os.path.join(TRUE_DIR, true_files[0]), delimiter=',').reshape(1,-1)
    trues = np.empty(shape=(len(true_files), TAKE*2))
    for i in range(len(true_files)):
        temp = np.genfromtxt(os.path.join(TRUE_DIR, true_files[i]), delimiter=',')
        trues[i, :] = np.append(temp[:TAKE,0], temp[:TAKE,1], axis=0)
    falses = np.empty(shape=(len(false_files),TAKE*2))
    for i in range(len(false_files)):
        temp = np.genfromtxt(os.path.join(FALSE_DIR, false_files[i]), delimiter=',')
        falses[i, :] = np.append(temp[:TAKE,0], temp[:TAKE,1], axis=0)

    totes = np.vstack((trues, falses)).reshape((-1,TAKE*2), order='F')
    totes_labels = np.empty(shape=(totes.shape[0], 2))
    totes_labels[0::2, :] = [1,0]
    totes_labels[1::2, :] = [0,1]

    train_im = totes[:-100,:]
    train_lab = totes_labels[:-100,:]

    test_im = totes[-100:,:]
    test_lab = totes_labels[-100:,:]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allocator_type = 'BFC'
    config.log_device_placement = True

    total_iterations = 50000
    epoch_size = 50
    num_steps = total_iterations // epoch_size
    batch_size = 50

    with tf.device('/cpu:0'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, TAKE*2])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 2])

        # Build the graph for the deep net
        # net, keep_prob = deepnn(x)
        net, keep_prob = cpcpff(x)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net['y_conv']))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(net['y_conv'], 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        num_epoch = 0
        sess.run(tf.global_variables_initializer())
        for i in range(total_iterations):
            batch_im = get_batch(batch_size, i, train_im)
            batch_lab = get_batch(batch_size, i, train_lab)
            if i % epoch_size == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_im,
                                                        y_: batch_lab, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={x: batch_im, y_: batch_lab, keep_prob: 0.7})
            num_epoch = num_epoch + 1

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_im, y_: test_lab, keep_prob: 1.0}))

        print('saving ...')
        save_path = saver.save(sess, os.path.join(SAVE_PATH, model_name+'.ckpt'))
        print("model saved in file: {}".format(save_path))

def deepnn(x):

   with tf.device('/cpu:0'):
       ret = {}
       num_filters = 32
       ret['input'] = tf.reshape(x, [-1, TAKE, 2, 1], name='input')
       ret['W_conv1'] = weight_variable([4, 2, 1, num_filters])
       ret['b_conv1'] = bias_variable([num_filters])
       ret['h_conv1'] = tf.nn.relu(conv2d(ret['input'], ret['W_conv1']) + ret['b_conv1'])

       tf.summary.scalar('W_conv1', ret['W_conv1'])
       ret['h_conv1_reshaped'] = tf.reshape(ret['h_conv1'], [-1, TAKE*2*num_filters])

    #    ret['input_reshaped'] = tf.reshape(ret['input'], [-1,TAKE*2])
       ret['W_fc1'] = weight_variable([TAKE*2*num_filters, 1024])
       ret['b_fc1'] = bias_variable([1024])

       ret['h_fc1'] = tf.nn.relu(tf.matmul(ret['input_reshaped'], ret['W_fc1']) + ret['b_fc1'])
    #    ret['h_fc1'] = tf.nn.relu(tf.matmul(ret['h_conv1_reshaped'], ret['W_fc1']) + ret['b_fc1'])

       # Dropout - controls the complexity of the model, prevents co-adaptation of
       # features.
       keep_prob = tf.placeholder(tf.float32)
       ret['h_fc1_drop'] = tf.nn.dropout(ret['h_fc1'], keep_prob)

       # Map the 1024 features to 10 classes, one for each digit
       ret['W_fc2'] = weight_variable([1024, 2])
       ret['b_fc2'] = bias_variable([2])

       ret['y_conv'] = tf.matmul(ret['h_fc1_drop'], ret['W_fc2']) + ret['b_fc2']
       return ret, keep_prob

def cpcpff(x):

    with tf.device('/cpu:0'):
        ret = {}

        ret['input'] = tf.reshape(x, [-1, TAKE, 2, 1], name='input')

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        ret['W_conv1'] = weight_variable([4, 2, 1, 32])
        ret['b_conv1'] = bias_variable([32])
        ret['h_conv1'] = tf.nn.relu(conv2d(ret['input'], ret['W_conv1']) + ret['b_conv1'])

        # Pooling layer - downsamples by 2X.
        ret['h_pool1'] = max_pool_2x2(ret['h_conv1'])

        # Second convolutional layer -- maps 32 feature maps to 64.
        ret['W_conv2'] = weight_variable([4, 2, 32, 64])
        ret['b_conv2'] = bias_variable([64])
        ret['h_conv2'] = tf.nn.relu(conv2d(ret['h_conv1'], ret['W_conv2']) + ret['b_conv2'])

        # Second pooling layer.
        ret['h_pool2'] = max_pool_2x2(ret['h_conv2'])

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        ret['W_fc1'] = weight_variable([TAKE//2 * 64, 1024])
        ret['b_fc1'] = bias_variable([1024])
        ret['h_conv2_flat'] = tf.reshape(ret['h_pool2'], [-1, TAKE//2*64])
        # ret['W_fc1_flat'] = tf.reshape(ret['W_fc1'], [-1, TAKE//4*64])
        ret['h_fc1'] = tf.nn.relu(tf.matmul(ret['h_conv2_flat'], ret['W_fc1']) + ret['b_fc1'])



        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        keep_prob = tf.placeholder(tf.float32)
        ret['h_fc1_drop'] = tf.nn.dropout(ret['h_fc1'], keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        ret['W_fc2'] = weight_variable([1024, 2])
        ret['b_fc2'] = bias_variable([2])

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
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='the name to attach to saved model', type=str)

    args = parser.parse_args()

    run(args.model_name)
