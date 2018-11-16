from tf_activation.l2_attack import CarliniL2
from tf_activation.models import carlini_attack_model as cam

import argparse
import sys
import os
import pickle

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer

SAVE_PATH = '../logdir/models'
model_name = 'carlini_attack.ckpt'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

TARGETED = True
LEARNING_RATE = 1e-10
MAX_ITERATIONS = 1000
BOXMAX = 1.0
BOXMIN = 0.0
ABORT_EARLY = True
CONFIDENCE = 1e-2

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


def generate_data(test_data, test_labels, samples, targeted=True, start=0, inception=False):
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
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(np.reshape(test_data[start+i], [28,28,-1]))
                # inputs.append(test_data[i+start])
                t = np.zeros(test_labels.shape[1])
                t[j] = 1
                targets.append(t)
        else:
            inputs.append(np.reshape(test_data[start+i], [28,28,-1]))
            targets.append(test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

inputs, targets = generate_data(mnist.test.images[1:2], mnist.test.labels[1:2], 1, targeted=TARGETED)

with tf.Session() as sess:

    with tf.device('/gpu:0'):

        model = cam.MNISTModel(session=sess, total_iterations=10000)
        # model.train()
        model.sess.run(tf.global_variables_initializer())
        model.restore('mnist_cff_2000.ckpt')

        attack = CarliniL2(model.sess, model, targeted=TARGETED, batch_size=9, learning_rate=LEARNING_RATE,
                            max_iterations=MAX_ITERATIONS, boxmax=BOXMAX, boxmin=BOXMIN,
                            abort_early=ABORT_EARLY, confidence = CONFIDENCE)
        # model.restore('mnist_cff_2000.ckpt')



        ret = attack.attack(inputs, targets)

        print("Model Accuracy: ", model.accuracy.eval(feed_dict={model.x:mnist.test.images[:100], model.y_:mnist.test.labels[:100], model.keep_prob:1.0}))

        for i in range(inputs.shape[0]):
            print(np.argmax(model.predict(tf.convert_to_tensor(np.reshape(inputs[i],[28,28,1]))).eval()))
            print(model.accuracy.eval(feed_dict={model.x: np.reshape(inputs[i], [-1,784]),
                                                    model.y_: np.reshape(targets[i], [-1,10])}))
            print(targets[i])


        print("Advs:")
        for i in range(ret.shape[0]):
            print(np.argmax(model.predict(tf.convert_to_tensor(ret[i,:,:,:])).eval()))
            # plt.imshow(ret[i,:,:,0], interpolation="nearest", cmap="gray")
            # plt.show()

    np.save('../logdir/adversaries/carlini_attacks.npy', ret)

    # for i in range(ret):
    #     show(ret[i])

# model.train()
