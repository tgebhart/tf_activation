from tf_activation.models import mnist_cff as mnist_model
from tf_activation.models.mnist_map import mnist_map

import math
import random
import os
import argparse
import errno
import itertools
from functools import wraps
import signal
import time

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
DIAGRAM_DATA_DIR_NAME = 'diagram_data'

S = 100
I = 0
P = 99

persistence_module = tf.load_op_library('/home/gebha095/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')

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

    plt.savefig(os.path.join(n, 'regression_' + str(fname) + '.svg'), dpi=1200,
                            format='svg', bbox_inches='tight')

def plot_diagram(diag, n, model):

    ax = plt.subplot()

    ax.scatter(diag[:,0], diag[:,1], s=25, c=diag[:,0]**2 - diag[:,1], cmap=plt.cm.coolwarm, zorder=10)
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

    plt.savefig(os.path.join(n, 'diagram_' + model + '.svg'), dpi=1200,
                            format='svg', bbox_inches='tight')

    plt.close()
    plt.clf()
    plt.cla()



def run(models, s, n=None, p=P):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    if n is None:
        n = os.path.join(DATA_PATH, time.strftime("%H:%M:%S_%d-%m-%y"))
    else:
        n = os.path.join(DATA_PATH, n)
    if not os.path.exists(n):
        os.makedirs(n)
    if not os.path.exists(os.path.join(n, DIAGRAM_DATA_DIR_NAME)):
        os.makedirs(os.path.join(n, DIAGRAM_DATA_DIR_NAME))

    images = mnist.test.images[:s]
    labels = mnist.test.labels[:s]

    diagram_data_dir = os.path.join(n, DIAGRAM_DATA_DIR_NAME)

    # permutations = itertools.permuations(models, 2)

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

    with tf.Session() as sess:

        for model in models:

            saver.restore(sess, os.path.join(SAVE_PATH, model))

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

            for i in range(0,images.shape[0]-1):

                ps1 = percentiles.eval(feed_dict={x: images[i:i+1], keep_prob:1.0})

                diagram_filename = os.path.join(diagram_data_dir, 'diagram_' + model + str(i) + '.csv')

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
                                                                    ps1,
                                                                    diagram_filename
                                                                    )
                r = result.eval(feed_dict={x: images[i:i+1], keep_prob:1.0})

                diag = np.genfromtxt(diagram_filename, delimiter=',')

                plot_diagram(diag, n, model)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', nargs='+', help='list of models to compare', required=True)
    parser.add_argument('-s', '--steps', help='number of images to compare', type=int, required=True)
    parser.add_argument('-n', '--folder_name', help='name of the folder holding image results', type=str)
    parser.add_argument('-p', '--percentile', help='filtration percentile above which to take edge weights', type=float, default=P)

    args = parser.parse_args()

    run(args.models, args.steps, n=args.folder_name, p=args.percentile)
