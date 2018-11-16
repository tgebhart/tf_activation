
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

# kmeans = KMeans(init='k-means++', n_clusters=len(mnist.train.labels[0]), n_init=10)
# kmeans = KMeans(init='k-means++', n_clusters=50, n_init=10)
# kmeans.fit(mnist.train.images)
# joblib.dump(kmeans, os.path.join(SAVE_PATH, 'kmeans10.pkl'))
kmeans = joblib.load(os.path.join(SAVE_PATH, 'kmeans10.pkl'))

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

@timeout(10)
def per_distance_func(result):
    per_distance = result.eval(feed_dict={x: test_inputs, keep_prob:1.0})
    return per_distance

df = []
centers = kmeans.cluster_centers_
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

    for i in range(NUM_STEPS):
        col = {}
        s = kmeans.score(mnist.train.images[i].reshape(1,-1))
        c = kmeans.predict(mnist.train.images[i].reshape(1,-1))[0]
        d = np.linalg.norm(c - mnist.train.images[i])
        # center_num = np.where(np.all(c == centers, axis=0))[0][0]
        col['score'] = s
        col['distance'] = d
        col['correct'] = np.argmax(mnist.train.labels[i])
        col['center'] = c

        center_corr = np.zeros(shape=mnist.train.labels[i].shape)
        center_corr[c] = 1
        test_inputs = np.stack((mnist.train.images[i], centers[c]))
        test_labels = np.stack((mnist.train.labels[i], center_corr))

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

        try:
            per_distance = per_distance_func(result)
        except TimeoutError:
            print('TimeoutError!')
            timeouts.append(i)
            continue

        print('Test Image: {}, Persistence Distance: {}'.format(i, per_distance))
        ce = cross_entropy.eval(feed_dict={x:test_inputs, y_:test_labels, keep_prob:1.0})
        y_conv = sess.run(net['y_conv'], feed_dict={x:test_inputs, keep_prob:1.0})
        acc = accuracy.eval(feed_dict={x:test_inputs, y_:test_labels, keep_prob:1})
        acc_solo = accuracy.eval(feed_dict={x:test_inputs[:1], y_:test_labels[:1], keep_prob:1})
        y_conv = y_conv / np.linalg.norm(y_conv)
        col['per_distance'] = per_distance[0]
        col['cross_entropy'] = ce
        col['y_conv'] = y_conv[0,np.argmax(test_labels[1], axis=0)]
        col['accuracy'] = acc
        col['acc_solo'] = acc_solo

        df.append(col)

        pdf = pd.DataFrame(df)
        pdf.to_pickle(os.path.join(RESULT_PATH, 'cluster_df.pkl'))
