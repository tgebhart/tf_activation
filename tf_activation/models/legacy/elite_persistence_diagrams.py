from tf_activation import DeepGraph
from tf_activation.models import mnist_cff as mnist_model
from tf_activation.experiments import distance_finder
from tf_activation.models.mnist_map import mnist_map

import math
import random
import os
from os import listdir
from os.path import isfile

import networkx as nx
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

persistence_module = tf.load_op_library('/home/tgebhart/python/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allocator_type = 'BFC'
config.log_device_placement = False

model = 'mnist_cff_2000.ckpt'
SAVE_PATH = '../../logdir/models'
ELITE_LOC = '../../logdir/elites/mnist_test_elites_19:45:10_18-09-17'
DIAG_DIR = os.path.join(ELITE_LOC, 'diagrams')

if not os.path.exists(DIAG_DIR):
    os.makedirs(DIAG_DIR)
ED_DIRECTORY = os.path.join(DIAG_DIR, 'elites')
if not os.path.exists(ED_DIRECTORY):
    os.makedirs(ED_DIRECTORY)
TD_DIRECTORY = os.path.join(DIAG_DIR, 'true')
if not os.path.exists(TD_DIRECTORY):
    os.makedirs(TD_DIRECTORY)

csvs = [f for f in listdir(ELITE_LOC) if isfile(os.path.join(ELITE_LOC, f))]
p = 99

def break_filename(f):
    cls = int(f[0])
    pidx = f.index('.')
    uidx = f.index('_')
    idx = int(f[uidx+1:pidx])
    fname = f[:pidx]
    return cls, idx, fname

with tf.device('/cpu:0'):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    net, keep_prob = mnist_model.deepnn(x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net['y_conv']))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(net['y_conv'], 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

df = []

with tf.Session(config=config) as sess:

    saver.restore(sess, os.path.join(SAVE_PATH, model))

    for i in range(len(csvs)):
        cls, idx, fname = break_filename(csvs[i])
        c = np.genfromtxt(os.path.join(ELITE_LOC, fname + '.csv'), delimiter=',')
        print(cls, idx, fname)
        col = {}
        col['correct'] = np.argmax(mnist.train.labels[idx])

        dfile = os.path.join(ED_DIRECTORY, fname + '.csv')
        dfile_o = os.path.join(TD_DIRECTORY, fname + '.csv')

        print(dfile)

        test_inputs = np.stack((mnist.train.images[idx], c))
        test_labels = np.stack((mnist.train.labels[idx], mnist.train.labels[idx]))

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

        ps1 = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})
        ps2 = percentiles.eval(feed_dict={x: test_inputs[1:2], keep_prob:1.0})

        resultF = persistence_module.input_graph_persistence([net['input'],
                                                            net['W_conv1'],
                                                            net['h_conv1'],
                                                            net['h_conv1'],
                                                            net['W_fc1'],
                                                            net['h_fc1'],
                                                            net['h_fc1_drop'],
                                                            net['W_fc2'],
                                                            net['y_conv']],
                                                            [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                            np.stack((ps1,ps2)), dfile)

        ceF = cross_entropy.eval(feed_dict={x:test_inputs[1:], y_:test_labels[1:], keep_prob:1.0})
        y_convF = sess.run(net['y_conv'], feed_dict={x:test_inputs[1:], keep_prob:1.0})
        accF = accuracy.eval(feed_dict={x:test_inputs[1:], y_:test_labels[1:], keep_prob:1})
        y_convF = y_convF / np.linalg.norm(y_convF)

        col['diagramF'] = resultF.eval(feed_dict={x:test_inputs[1:], y_:test_labels[1:], keep_prob:1.0})
        col['cross_entropyF'] = ceF
        col['y_convF'] = y_convF[0,np.argmax(test_labels[1], axis=0)]
        col['accuracyF'] = accF

        resultT = persistence_module.input_graph_persistence([net['input'],
                                                            net['W_conv1'],
                                                            net['h_conv1'],
                                                            net['h_conv1'],
                                                            net['W_fc1'],
                                                            net['h_fc1'],
                                                            net['h_fc1_drop'],
                                                            net['W_fc2'],
                                                            net['y_conv']],
                                                            [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                            np.stack((ps1,ps2)), dfile_o)

        ceT = cross_entropy.eval(feed_dict={x:test_inputs[0:1], y_:test_labels[0:1], keep_prob:1.0})
        y_convT = sess.run(net['y_conv'], feed_dict={x:test_inputs[0:1], keep_prob:1.0})
        accT = accuracy.eval(feed_dict={x:test_inputs[0:1], y_:test_labels[0:1], keep_prob:1})
        y_convT = y_convT / np.linalg.norm(y_convT)

        col['diagramT'] = resultT.eval(feed_dict={x:test_inputs[0:1], y_:test_labels[0:1], keep_prob:1.0})
        col['cross_entropyT'] = ceT
        col['y_convT'] = y_convT[0,np.argmax(test_labels[0], axis=0)]
        col['accuracyT'] = accT

        print(col['diagramF'].shape, col['diagramT'].shape)

        df.append(col)

df = pd.DataFrame(df)
df.to_pickle(DIAG_DIR)
