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
from functools import wraps
import errno
import os
import signal

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

p = 99
dp = 0.001

SAVE_PATH = '../logdir/models'
FIGURE_PATH = '../logdir/data/graphs/plots'
GRAPHML_PATH = '../logdir/data/graphs/percentiles/' + str(100*dp)
model = 'mnist_cff50.ckpt'

if not os.path.exists(GRAPHML_PATH):
    os.makedirs(GRAPHML_PATH)


mnist_map = {
    0: 3,
    1: 2,
    2: 1,
    3: 18,
    4: 4,
    5: 8,
    6: 11,
    7: 0,
    8: 61,
    9: 7
}

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
    for i in range(1000):
        print("i: {}".format(i))
        correct_label = np.argmax(mnist.train.labels[i])

        diagram_filename = os.path.join(GRAPHML_PATH, str(correct_label)+'_'+str(i)+'.graphml')

        saver.restore(sess, os.path.join(SAVE_PATH, model))

        test_inputs = np.stack((mnist.train.images[i], mnist.train.images[i]))
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
                                                                0,
                                                                diagram_filename
                                                                )


        print(result.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0}))


        # g = nx.read_graphml(diagram_filename)
        # nx.draw_spring(g, node_size=5)
        # plt.savefig(os.path.join(FIGURE_PATH, 'spring_'+str(correct_label)+'_'+str(i)+'.svg'), format='svg', dpi=1200)
        # plt.close()
