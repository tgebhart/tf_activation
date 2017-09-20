from tf_activation import DeepGraph
from tf_activation.models import mnist_cff as mnist_model
from tf_activation.models import mnist_adversary_generator as generator

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

MODEL = 'mnist_cff_2000.ckpt'
SAVE_FOLDER_NAME = 'mnist_test_adversaries_' + time.strftime("%H:%M:%S_%d-%m-%y")

TRIALS = 100000
SIZE = 250
NUM_CHANGES = 10
PRINT_EVERY = 100
EPSILON = .005

if not os.path.exists(os.path.join(SAVE_DIR, SAVE_FOLDER_NAME)):
    os.makedirs(os.path.join(SAVE_DIR, SAVE_FOLDER_NAME))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
model = os.path.join(MODEL_DIR, MODEL)

for i in range(mnist.test.images.shape[0]):

    im = mnist.test.images[i]
    lab = mnist.test.labels[i]
    label = np.argmax(lab)
    print('Creating Adversary for index {} with correct label {}'.format(i, label))

    adv = generator.train_adversaries(model, im, lab, trials=TRIALS, \
                                    print_every=PRINT_EVERY, size=SIZE, \
                                    num_changes=NUM_CHANGES, epsilon=EPSILON)

    if adv is not None:

        np.savetxt(os.path.join(SAVE_DIR, SAVE_FOLDER_NAME, str(label) + '_' + str(i) + '.csv'), \
                    adv, delimiter=',')
