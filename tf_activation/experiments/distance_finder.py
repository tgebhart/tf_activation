from tf_activation import DeepGraph
from tf_activation.models import mnist_cff as mnist_model
from tf_activation.models.mnist_map import mnist_map

import math
import random
import os
import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

SAVE_PATH = '../../logdir/models'
ADV_PATH = '../../logdir/adversaries'
ELITE_PATH = '../../logdir/elites'
DATA_PATH = '../../logdir/data/experiments'

S = 100
I = 0
P = 99

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.allow_soft_placement = True
config.gpu_options.allocator_type = 'BFC'
config.log_device_placement = True

persistence_module = tf.load_op_library('/home/tgebhart/python/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')

def create_path(start_im, end_im, steps=100):
    ret = []
    t = np.divide(end_im - start_im, steps)
    for i in range(1,steps+1):
        ret.append(start_im + i*t)
    return ret

def save_interpolation(im, step, n):
    plt.imsave(os.path.join(n, 'interpolation_' + str(step) + '.png'),
                np.reshape(im,[28,28]), cmap="gray")

def regress(X, y, xlab, ylab, n, fname=None):

    if fname is None:
        fname = xlab + '_' + ylab

    # Split the data into training/testing sets
    X_train = X[:-20]
    X_test = X[-20:]

    # Split the targets into training/testing sets
    y_train = y[:-20]
    y_test = y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # The coefficients
    print('Coefficients: ', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(X_test) - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X_test, y_test))
    print('R^2: %.2f' % r2_score(y_test, regr.predict(X_test)))

    fig, ax = plt.subplots()
    X_min = np.min(X)
    X_max = np.max(X)
    y_min = np.min(y)
    y_max = np.max(y)
    # Plot outputs

    ax.scatter(X, y,  color='black')
    ax.plot(X, regr.predict(X), color='silver', linewidth=3)
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title('{} vs {}'.format(xlab, ylab))

    plt.savefig(os.path.join(n, 'regression_' + str(fname) + '.png'))

def run(model, l=None, i=I, f=None, s=S, n=None, p=P, c=None, m=None, e=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    if l is None and m is None:
        print('Must specify last_image or mnist image to interpolate to!')
        sys.exit()
    if l is not None:
        iml = np.genfromtxt(l, delimiter=',')
    else:
        iml = mnist.test.images[mnist_map[m]]
    if n is None:
        n = os.path.join(DATA_PATH, time.strftime("%H:%M:%S_%d-%m-%y"))
    else:
        n = os.path.join(DATA_PATH, n)
    if not os.path.exists(n):
        os.makedirs(n)
    if f is not None:
        imf = np.genfromtxt(f, delimiter=',')
    else:
        imf = mnist.test.images[mnist_map[i]]
    if c is None:
        c = i
    if e is None:
        e = s

    test_inputs = np.stack((imf, iml))
    correct_label = mnist.test.labels[mnist_map[c]]
    test_labels = np.stack((correct_label, correct_label))

    columns = ['in_distance', 'per_distance', 'cross_entropy', 'y_conv', 'accuracy']
    index = range(s)
    test_df = pd.DataFrame(index=index, columns=columns)

    path = create_path(imf, iml)

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

        saver.restore(sess, os.path.join(SAVE_PATH, model))
        test_inputs = np.stack((imf, path[0]))

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

        for i in range(len(path)):

            test_inputs = np.stack((imf, path[i]))
            in_distance = np.linalg.norm(imf - path[i], ord=2)

            if i % e == 0:
                save_interpolation(path[i], i, n)

            ps2 = percentiles.eval(feed_dict={x: test_inputs[1:2], keep_prob:1.0})

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
                                                            np.stack((ps1, ps2))
                                                            )

            per_distance = result.eval(feed_dict={x: test_inputs, keep_prob:1.0})
            print('Step: ', i)
            print('distance:', per_distance)

            ce = cross_entropy.eval(feed_dict={x:test_inputs[1:], y_:test_labels[1:], keep_prob:1.0})
            y_conv = sess.run(net['y_conv'], feed_dict={x:test_inputs[1:], keep_prob:1.0})
            acc = accuracy.eval(feed_dict={x:test_inputs[1:], y_:test_labels[1:], keep_prob:1})
            y_conv = y_conv / np.linalg.norm(y_conv)

            test_df.loc[i, 'in_distance'] = in_distance
            test_df.loc[i, 'per_distance'] = per_distance[0]
            test_df.loc[i, 'cross_entropy'] = ce
            test_df.loc[i, 'y_conv'] = y_conv[0,np.argmax(test_labels[1], axis=0)]
            test_df.loc[i, 'accuracy'] = acc

    X = test_df['in_distance'].as_matrix()
    X = X.reshape((X.shape[0], 1))
    y = test_df['per_distance'].as_matrix()
    regress(X, y, 'Input Distance', 'Persistence Distance', n)

    X = test_df['cross_entropy'].as_matrix()
    X = X.reshape((X.shape[0], 1))
    regress(X, y, 'Cross-Entropy', 'Persistence Distance', n)

    X = test_df['y_conv'].as_matrix()
    X = X.reshape((X.shape[0], 1))
    regress(X, y, 'Correct Class Probability', 'Persistence Distance', n)

    X = test_df['in_distance'].as_matrix()
    X = X.reshape((X.shape[0], 1))
    y = test_df['y_conv'].as_matrix()
    regress(X, y, 'Input Distance', 'Correct Class Probability', n)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='the model to use', type=str)
    parser.add_argument('-i', '--integer', help='image of mnist dataset', type=int, default=I)
    parser.add_argument('-f', '--first_image', help='full path of starting image (if not using mnist image)', type=str)
    parser.add_argument('-s', '--int_steps', help='number of interpolation steps', type=int, default=S)
    parser.add_argument('-n', '--folder_name', help='name of the folder holding image results', type=str)
    parser.add_argument('-p', '--percentile', help='filtration percentile above which to take edge weights', type=float, default=P)
    parser.add_argument('-c', '--correct_label', help='correct class label for the starting image', type=int)
    parser.add_argument('-l', '--last_image', help='full path of the ending image to be interpolated to', type=str)
    parser.add_argument('-m', '--mnist_last', help='the integer of the mnist dataset if to be used as `last_image`', type=int)
    parser.add_argument('-e', '--save_every', help='number of runs between saving each interpolated image', type=int)

    args = parser.parse_args()

    run(args.model, l=args.last_image, i=args.integer, f=args.first_image,
        s=args.int_steps, n=args.folder_name, p=args.percentile, c=args.correct_label,
        m=args.mnist_last, e=args.save_every)
