from tf_activation import DeepGraph
from tf_activation.models import mnist_cff as mnist_model

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
SAVE_DIR = '../../logdir/elites'

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

TRIALS = 1000
SIZE = 200
NUM_CHANGES = 10
PRINT_EVERY = 100

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

def generate(model, i=None, lab=None, trials=None, size=None, num_changes=None, print_every=None, save_name=None):

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
    if num_changes is None:
        num_changes = NUM_CHANGES
    if print_every is None:
        print_every = PRINT_EVERY
    if save_name is None:
        integer = get_nonzero(lab)
        save_name = 'mnist_' + str(integer) + '_' + time.strftime("%H:%M:%S_%d-%m-%y") + '.csv'

    elite = train_elites(model, im, lab, trials=trials, size=size, num_changes=num_changes, print_every=print_every)

    np.savetxt(os.path.join(SAVE_DIR, save_name), elite, delimiter=',')

def get_nonzero(arr):
    for i in range(len(arr)):
        if arr[i] == 1:
            return i


def build_elites(im, size=200, num_changes=100):
    elites = np.empty(shape=(size+1,im.shape[0],im.shape[1]))
    for i in range(size):
        idxs = np.random.choice(im.shape[1], size=num_changes)
        elite = np.copy(im[0,:])
        for idx in idxs:
            elite[idx] = elite[idx] + random.uniform(-elite[idx], 1-elite[idx])
        elites[i,:,:] = elite
    elites[size,:,:] = im
    return elites


def train_elites(model, im, lab, trials=10000, size=200, num_changes=10, print_every=100):

    saver = tf.train.Saver()
    ret_elite = None
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(MODEL_DIR, model))
        im = np.reshape(im, (1,im.shape[0]))
        lab = np.reshape(lab, (1, lab.shape[0]))
        print('Initial error: {}'.format(cross_entropy.eval(feed_dict={x: im, y_:lab, keep_prob: 1.0})))

        trial = 0
        trial_im = im
        while trial < trials:
            elite_set = build_elites(trial_im, size=size, num_changes=num_changes)
            results = np.zeros(shape=(elite_set.shape[0]))
            for i in range(elite_set.shape[0]):
                results[i] = cross_entropy.eval(feed_dict={x: elite_set[i,:, :], y_:lab, keep_prob: 1.0})
            min_idx = np.argmin(results)
            if trial % print_every == 0:
                print('Min cross-entropy in trial {}: {}'.format(trial, results[min_idx]))
            trial_im = elite_set[min_idx,:,:]
            if results[min_idx] == 0:
                print('breaking out, elite found')
                break
            trial = trial + 1
        ret_elite = trial_im

    return ret_elite


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='the model to use', type=str)
    parser.add_argument('-i', '--integer', help='image of mnist dataset', type=int)
    # parser.add_argument('-l', '--label', help='label of mnist dataset', type=int)
    parser.add_argument('-t', '--trials', help='number of trials', type=int)
    parser.add_argument('-s', '--size', help='size of elites generated per trial', type=int)
    parser.add_argument('-n', '--num_changes', help='number of pixels to change per trial', type=int)
    parser.add_argument('-p', '--print_every', help='print after every `print_every` trials', type=int)
    parser.add_argument('-a', '--all', help='run all digits', type=bool)

    args = parser.parse_args()

    if args.all:
        for k in mnist_map.keys():
            generate(args.model, i=k, lab=k, trials=args.trials, size=args.size, num_changes=args.num_changes, print_every=args.print_every)
    else:
        generate(args.model, i=args.integer, lab=args.integer, trials=args.trials, size=args.size, num_changes=args.num_changes, print_every=args.print_every)
