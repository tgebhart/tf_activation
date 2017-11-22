from tf_activation import DeepGraph
from tf_activation.models import mnist_cff as mnist_model

import math
import random
import os
from os import listdir
from os.path import isfile, join

import networkx as nx
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score

from tensorflow.examples.tutorials.mnist import input_data

CLUSTER_PATH = '../../logdir/data/experiments/cluster_mnist'
SAVE_PATH = '../../logdir/models'
ADV_PATH = '../../logdir/adversaries/carlini_attacks_targeted5020'
ELITE_PATH = '../../logdir/elites'
DATA_PATH = '../../logdir/data'
SAVE_FIG_LOC = '../../logdir/figures'
MODEL = 'mnist_cff50.ckpt'

centers = np.load(os.path.join(SAVE_PATH, 'average_mnist.npy'))
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
advs = [f for f in listdir(ADV_PATH) if isfile(join(ADV_PATH, f))]
distance_stats = pd.read_pickle(os.path.join(SAVE_PATH, 'distance_stats.pkl'))

config = tf.ConfigProto()

persistence_module = tf.load_op_library('/home/tgebhart/python/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')


# setup model
with tf.device('/cpu:0'):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    net, keep_prob = mnist_model.deepnn(x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net['y_conv']))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    predicted_class = tf.argmax(net['y_conv'], 1)
    correct_prediction = tf.equal(tf.argmax(net['y_conv'], 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()


    # run network
df = []
p = 99
h = 0
with tf.Session(config=config) as sess:

    saver.restore(sess, os.path.join(SAVE_PATH, MODEL))

    for adv in advs:
        print("Input file:", adv)
        col = {}

        im_adv = np.genfromtxt(join(ADV_PATH, adv), delimiter=',')
        t = int(adv[2])
        idx = int(adv[4:adv.find('.')])
        im_true = mnist.train.images[idx]

        c = centers[:,0]

        test_inputs = np.stack((im_adv, c))
        test_labels = np.stack((mnist.train.labels[idx],mnist.train.labels[idx]))

        ce = cross_entropy.eval(feed_dict={x:test_inputs[0:1], y_:test_labels[0:1], keep_prob:1.0})
        y_conv = sess.run(net['y_conv'], feed_dict={x:test_inputs[0:1], keep_prob:1.0})
        pc = np.argmax(y_conv[0,:])
        acc = accuracy.eval(feed_dict={x:test_inputs[0:1], y_:test_labels[0:1], keep_prob:1})
        y_conv = y_conv / np.linalg.norm(y_conv)

        col['cross_entropy'] = ce
        col['y_conv'] = y_conv
        col['predicted_class'] = pc
        col['accuracy'] = acc
        col['is_adv'] = True

        c = centers[:,pc]

        test_inputs = np.stack((im_adv, c))

        d_adv_true = np.linalg.norm(im_adv - im_true, ord=2)
        d_adv_c = np.linalg.norm(im_adv - c, ord=2)

        col['distance_to_average'] = d_adv_c
        col['distance_to_true'] = d_adv_true


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

        psa = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})
        psc = percentiles.eval(feed_dict={x: test_inputs[1:2], keep_prob:1.0})


        result = persistence_module.wasserstein_distance([net['input'],
                                                            net['W_conv1'],
                                                            net['h_conv1'],
                                                            net['h_conv1'],
                                                            net['W_fc1'],
                                                            net['h_fc1'],
                                                            net['h_fc1_drop'],
                                                            net['W_fc2'],
                                                            net['y_conv']],
                                                            [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                            np.stack((psa, psc)),
                                                            h
                                                            )


        per_distance = result.eval(feed_dict={x: test_inputs[:2], keep_prob:1.0})
        col['persistence_distance'] = per_distance[0]

        df.append(col)
        col = {}

        test_inputs = np.stack((im_true, c))
        test_labels = np.stack((mnist.train.labels[idx],mnist.train.labels[idx]))

        ce = cross_entropy.eval(feed_dict={x:test_inputs[0:1], y_:test_labels[0:1], keep_prob:1.0})
        y_conv = sess.run(net['y_conv'], feed_dict={x:test_inputs[0:1], keep_prob:1.0})
        pc = np.argmax(y_conv[0,:])
        acc = accuracy.eval(feed_dict={x:test_inputs[0:1], y_:test_labels[0:1], keep_prob:1})
        y_conv = y_conv / np.linalg.norm(y_conv)

        col['cross_entropy'] = ce
        col['y_conv'] = y_conv
        col['predicted_class'] = pc
        col['accuracy'] = acc
        col['is_adv'] = False

        c = centers[:,pc]

        test_inputs = np.stack((im_true, c))

        d_true_c = np.linalg.norm(im_true - c, ord=2)

        col['distance_to_average'] = d_true_c
        col['distance_to_true'] = 0

        pst = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})

        result = persistence_module.wasserstein_distance([net['input'],
                                                            net['W_conv1'],
                                                            net['h_conv1'],
                                                            net['h_conv1'],
                                                            net['W_fc1'],
                                                            net['h_fc1'],
                                                            net['h_fc1_drop'],
                                                            net['W_fc2'],
                                                            net['y_conv']],
                                                            [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                            np.stack((pst, psc)),
                                                            h
                                                            )


        per_distance = result.eval(feed_dict={x: test_inputs[:2], keep_prob:1.0})
        col['persistence_distance'] = per_distance[0]

        df.append(col)

df = pd.DataFrame(df)
df.to_pickle('../../logdir/data/experiments/adv_detection.pkl')
