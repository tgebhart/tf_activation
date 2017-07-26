from tf_activation import DeepGraph
from tf_activation.models import mnist_cpcpff as mnist_model

import math
import random
import argparse
import time
import os

import networkx as nx
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

MODEL_DIR = '../../logdir/models'
SAVE_DIR = '../../logdir/adversaries'

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

TRIALS = 10000
SIZE = 200
NUM_CHANGES = 10
PRINT_EVERY = 100
EPSILON = .005

seed = 1
random.seed(a=seed)
np.random.seed(seed=seed)

with tf.device('/cpu:0'):

    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    net, keep_prob = mnist_model.deepnn(x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net['y_conv']))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(net['y_conv'], 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def generate(model, i=None, lab=None, trials=None, size=None, epsilon=None, num_changes=None, print_every=None, save_name=None):

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    if i is None:
        im = mnist.test.images[0]
        i = 7
    else:
        im = mnist.test.images[mnist_map[i]]
    if lab is None:
        lab = mnist.test.labels[0]
    else:
        lab = mnist.test.labels[mnist_map[i]]
    if trials is None:
        trials = TRIALS
    if size is None:
        size = SIZE
    if epsilon is None:
        epsilon = EPSILON
    if num_changes is None:
        num_changes = NUM_CHANGES
    if print_every is None:
        print_every = PRINT_EVERY
    if save_name is None:
        save_name = 'mnist_' + str(i) + '_' + time.strftime("%H:%M:%S_%d-%m-%y") + '.csv'

    adversary = train_adversaries(model, im, lab, trials=trials, size=size, num_changes=num_changes, epsilon=epsilon, print_every=print_every)

    np.savetxt(os.path.join(SAVE_DIR, save_name), adversary, delimiter=',')

def get_nonzero(arr):
    for i in range(len(arr)):
        if arr[i] == 1:
            return i

def build_adversaries(im, size=200, num_changes=10, epsilon=.005):
    adversaries = np.empty(shape=(size,im.shape[0],im.shape[1]))
    for i in range(size):
        idxs = np.random.choice(im.shape[1], size=num_changes)
        adversary = np.copy(im[0,:])
        for idx in idxs:
            adversary[idx] = adversary[idx] + random.uniform(-epsilon, epsilon)
        adversaries[i,:,:] = adversary
    return adversaries


def train_adversaries(model, im, lab, trials=100000, print_every=100, size=200, num_changes=10, epsilon=.005):

    saver = tf.train.Saver()
    ret_adversary = None
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(MODEL_DIR, model))
        im = np.reshape(im, (1,im.shape[0]))
        lab = np.reshape(lab, (1, lab.shape[0]))
        print('Initial error: {}'.format(cross_entropy.eval(feed_dict={x: im, y_:lab, keep_prob: 1.0})))

        trial = 0
        trial_im = im
        while trial < trials:
            adversary_set = build_adversaries(trial_im, size=size, num_changes=num_changes, epsilon=epsilon)
            results = np.zeros(shape=(adversary_set.shape[0]))
            for i in range(adversary_set.shape[0]):
                results[i] = cross_entropy.eval(feed_dict={x: adversary_set[i,:, :], y_:lab, keep_prob: 1.0})
            max_idx = np.argmax(results)
            trial_im = adversary_set[max_idx,:,:]
            acc = accuracy.eval(feed_dict={x: trial_im,y_: lab, keep_prob: 1.0})
            if trial % print_every == 0:
                print('Max cross-entropy in trial {}: {}'.format(trial, results[max_idx]))
                print('Accuracy at trial {}: {}'.format(trial,  acc))
            if acc != 1:
                ret_adversary = trial_im
                print('FOUND ADVERSARY AT TRIAL {}'.format(trial))
                print('CLASSIFIED AS: {}'.format(np.argmax(net['y_conv'].eval(feed_dict={x: trial_im,y_: lab, keep_prob: 1.0}))))
                break
            trial = trial + 1

    if ret_adversary is None:
        print('DID NOT CONVERGE AFTER {} trials'.format(trial))
    return ret_adversary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='the model to use', type=str)
    parser.add_argument('-i', '--integer', help='image of mnist dataset', type=int)
    # parser.add_argument('-l', '--label', help='label of mnist dataset', type=int)
    parser.add_argument('-t', '--trials', help='number of trials', type=int)
    parser.add_argument('-s', '--size', help='size of elites generated per trial', type=int)
    parser.add_argument('-n', '--num_changes', help='number of pixels to change per trial', type=int)
    parser.add_argument('-e', '--epsilon', help='change around each pixel', type=float)
    parser.add_argument('-p', '--print_every', help='print after every `print_every` trials', type=int)
    parser.add_argument('-a', '--all', help='run all digits', type=bool)

    args = parser.parse_args()

    if args.all:
        for k in mnist_map.keys():
            generate(args.model, i=k, lab=k, trials=args.trials, size=args.size, num_changes=args.num_changes, epsilon=args.epsilon, print_every=args.print_every)
    else:
        generate(args.model, i=args.integer, lab=args.integer, trials=args.trials, size=args.size, num_changes=args.num_changes, epsilon=args.epsilon, print_every=args.print_every)
