from tf_activation.l2_attack2 import CarliniL2
from tf_activation.models import carlini_keras_attack_model as cam

import argparse
import sys
import os
import pickle

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer

SAVE_PATH = '../logdir/models'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

TARGETED = True
LEARNING_RATE = 1e-9
MAX_ITERATIONS = 10000
BOXMAX = 1.0
BOXMIN = 0.0
ABORT_EARLY = True
CONFIDENCE = 20

NUM_IMS = 50

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(test_data, test_labels, samples, targeted=True, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            seq = range(test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(test_labels[i])) and (inception == False):
                    continue
                inputs.append(np.reshape(test_data[i], [28,28,-1]))
                # inputs.append(test_data[i+start])
                t = np.zeros(test_labels.shape[1])
                t[j] = 1.0
                targets.append(t)
        else:
            inputs.append(np.reshape(test_data[i], [28,28,-1]))
            targets.append(test_labels[i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

inputs, targets = generate_data(mnist.test.images[:NUM_IMS], mnist.test.labels[:NUM_IMS], NUM_IMS, targeted=TARGETED)

with tf.Session() as sess:

    model = cam.MNISTModel('../logdir/models/keras_mnist_cff', session=sess)

    print(inputs.shape, targets.shape)

    attack = CarliniL2(sess, model, targeted=TARGETED, batch_size=9,
                        max_iterations=MAX_ITERATIONS, boxmax=BOXMAX, boxmin=BOXMIN,
                        abort_early=ABORT_EARLY, confidence=CONFIDENCE)

    ret = attack.attack(inputs, targets)

    # for i in range(len(ret)):
    #     print("Valid:")
    #     show(inputs[i])
    #     print("Adversarial:")
    #     show(ret[i])
    #
    #     print("Classification:", model.model.predict(ret[i:i+1]))
    #
    #     print("Total distortion:", np.sum((ret[i]-inputs[i])**2)**.5)

    if TARGETED:
        targeted_string = '_targeted'
    else:
        targeted_string = ''

    np.save('../logdir/adversaries/carlini_attacks' + targeted_string + str(NUM_IMS) + str(CONFIDENCE) + '.npy', ret)
