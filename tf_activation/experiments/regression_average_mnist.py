
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
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
persistence_module = tf.load_op_library('/home/tgebhart/python/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')

from time import time
import os
from functools import wraps
import errno
import os
import signal
import pickle

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

SAVE_PATH = '../../logdir/models'
RESULT_PATH = '../../logdir/data/experiments/cluster_mnist'
model = 'mnist_cff_2000.ckpt'
NUM_STEPS = 5000
p = 99

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.allow_soft_placement = True
config.gpu_options.allocator_type = 'BFC'
config.log_device_placement = False

centers = np.load(os.path.join(SAVE_PATH, 'average_mnist.npy'))

def per_distance_func(result):
    per_distance = result.eval(feed_dict={x: test_inputs, keep_prob:1.0})
    return per_distance

df = []
timeouts = []
with tf.device('/cpu:0'):

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    net, keep_prob = mnist_model.deepnn(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net['y_conv']))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(net['y_conv'], 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

with tf.Session(config=config) as sess:

    saver.restore(sess, os.path.join(SAVE_PATH, model))

    for i in range(1039,NUM_STEPS):

        if len(df) != i:
            try:
                with open(os.path.join(RESULT_PATH, 'intermediate_df.pkl'), 'rb') as f:
                    df = pickle.load(f)
            except Exception:
                pass

        col = {}
        correct = np.argmax(mnist.train.labels[i])
        c = centers[:,correct]
        d = np.linalg.norm(c - mnist.train.images[i])
        col['distance'] = d
        col['correct'] = correct

        test_inputs = np.stack((mnist.train.images[i], c))
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
                                                            [0,1,2,2,1,4,4,1,4],
                                                            [p,p,p])

        ps1 = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})
        ps2 = percentiles.eval(feed_dict={x: test_inputs[1:2], keep_prob:1.0})

        print('STEP:', i)
        print('Length Dataframe:', len(df))
        result = persistence_module.wasserstein_distance([net['input'],
                                                         net['W_conv1'],
                                                         net['h_conv1'],
                                                         net['h_conv1'],
                                                         net['W_fc1'],
                                                         net['h_fc1'],
                                                         net['h_fc1_drop'],
                                                         net['W_fc2'],
                                                         net['y_conv']],
                                                         [0,1,2,2,1,4,4,1,4],
                                                         np.stack((ps1,ps2)))

        per_distance = per_distance_func(result)

        print('Test Image: {}, Persistence Distance: {}'.format(i, per_distance))
        ce = cross_entropy.eval(feed_dict={x:test_inputs, y_:test_labels, keep_prob:1.0})
        y_conv = sess.run(net['y_conv'], feed_dict={x:test_inputs, keep_prob:1.0})
        acc = accuracy.eval(feed_dict={x:test_inputs, y_:test_labels, keep_prob:1})
        y_conv = y_conv / np.linalg.norm(y_conv)
        col['per_distance'] = per_distance[0]
        col['cross_entropy'] = ce
        col['y_conv'] = y_conv[0,np.argmax(test_labels[1], axis=0)]
        col['accuracy'] = acc

        df.append(col)

        with open(os.path.join(RESULT_PATH, 'intermediate_df.pkl'), 'wb') as f:
            pickle.dump(df, f)

    pdf = pd.DataFrame(df)
    pdf.to_pickle(os.path.join(RESULT_PATH, 'average_df.pkl'))
