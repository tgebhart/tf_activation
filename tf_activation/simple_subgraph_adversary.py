from tf_activation.models import mnist_cff as mnist_model
from tf_activation.experiments import distance_finder
from tf_activation.models.mnist_map import mnist_map

from sklearn.cluster import KMeans
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
persistence_module = tf.load_op_library('/home/tgebhart/python/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')

from time import time
import os
from os import listdir
from os.path import isfile, join
from functools import wraps
import errno
import os
import signal

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

SAVE_PATH = '../logdir/models'
ADV_SET = 'carlini_attacks_targeted1000'
FIGURE_PATH = '../logdir/data/graphs/adversaries/plots/' + ADV_SET
GRAPHML_PATH = '../logdir/data/graphs/adversaries/' + ADV_SET
ADV_PATH = '../logdir/adversaries/' + ADV_SET
model = 'mnist_cff50.ckpt'
advs = [f for f in listdir(ADV_PATH) if isfile(join(ADV_PATH, f))]
p = 99
dp = 85

config = tf.ConfigProto()
config.log_device_placement = False

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

with tf.Session(config=config) as sess:
    # advs = advs[:advs.index('9_7_16.csv')+1]
    for adv in advs:
        print("Adversary File: {}".format(adv))
        i = int(adv[4:adv.find('.')])

        correct_label = np.argmax(mnist.train.labels[i])

        diagram_filename = os.path.join(GRAPHML_PATH, adv[:adv.find('.')]+'.graphml')

        saver.restore(sess, os.path.join(SAVE_PATH, model))

        a = np.genfromtxt(join(ADV_PATH, adv), delimiter=',')

        test_inputs = np.stack((a, mnist.train.images[i]))
        test_labels = np.stack((mnist.train.labels[i], mnist.train.labels[i]))



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
        ps2 = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})

        result = persistence_module.persistent_sub_graph([net['input'],
                                                                net['W_conv1'],
                                                                net['h_conv1'],
                                                                net['h_conv1'],
                                                                net['W_fc1'],
                                                                net['h_fc1'],
                                                                net['h_fc1_drop'],
                                                                net['W_fc2'],
                                                                net['y_conv']],
                                                                [0, 1, 2, 2, 1, 4, 4, 1, 4],
                                                                np.stack((ps1,ps2)),
                                                                dp,
                                                                diagram_filename
                                                                )


        print(result.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0}))


        # g = nx.read_graphml(diagram_filename)
        # nx.draw_spring(g, node_size=5)
        # plt.savefig(os.path.join(FIGURE_PATH, 'spring_'+str(correct_label)+'_'+str(i)+'.svg'), format='svg', dpi=1200)
        # plt.close()
