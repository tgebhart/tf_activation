from tf_activation.models.cff import CFF_Model
from tf_activation.models.fff import FFF_Model
from tf_activation.models.ccff import CCFF_Model

from foolbox.models import TensorFlowModel
from foolbox.attacks import LBFGSAttack
from foolbox.criteria import TargetClassProbability
import foolbox

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import argparse
import os

from tensorflow.examples.tutorials.mnist import input_data


UPTO = 1000

def run(output_location, model_directory, trained_model, classes, upto=UPTO):

    model = CFF_Model(W_conv1=[5,5,1,32])

    model.infer_graph_parameters_from_filename(model_directory)
    training_parameters = model.infer_training_parameters_from_filename(trained_model)

    model_string = model_directory[model_directory.rfind('/')+1:]
    train_string = trained_model[:trained_model.rfind('/')]

    adversary_directory = os.path.join(output_location, model_string, train_string)

    config = tf.ConfigProto()
    config.log_device_placement = False

    with tf.device('/gpu:0'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        net = model.build_graph(x)
        if not model.implements_dropout:
            keep_prob = tf.placeholder(tf.float32)

        logits = model.get_output_layer()
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
        train_step = model.get_train_step(cross_entropy, training_parameters['optimizer'], lr=training_parameters['learning_rate'])
        correct_prediction = tf.equal(tf.argmax(model.get_output_layer(), 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        saver.restore(sess, os.path.join(model_directory, trained_model))

        X = mnist.test.images
        Y = mnist.test.labels

        for c in classes:

            criterion = TargetClassProbability(c, p=0.99)

            model = foolbox.models.TensorFlowModel(x, logits, (0, 1))
            attack = LBFGSAttack(model, criterion)

            adversary_subdirectory = os.path.join(adversary_directory, str(c))
            if not os.path.exists(adversary_subdirectory):
                os.makedirs(adversary_subdirectory)

            for i in range(upto):

                print('class: {}, example: {}'.format(c, i))

                label = np.argmax(Y[i])
                adversarial = attack(X[i], label=label)

                savepath = os.path.join(adversary_subdirectory, 'adversary_{}_correct_{}.npy'.format(i, label))
                np.save(savepath, adversarial)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--output_location",
                      required=True,
                      help="directory in which to store adversaries")
    parser.add_argument("-m","--model_directory",
                      required=True,
                      help="directory location of trained model to produce adversaries")
    parser.add_argument("-t","--trained_model",
                      required=True,
                      help="name of trained model checkpoint")
    parser.add_argument('-c','--classes',
                        required=True,
                        nargs='+',
                        help='List of classes to approximate adversarially')
    parser.add_argument('-ut','--upto',
                        type=int,
                        required=False,
                        default=UPTO,
                        help="Number examples to create persistent subgraphs from")


    args = parser.parse_args()
    classes = list(map(lambda x: int(x), args.classes))
    run(args.output_location, args.model_directory, args.trained_model, classes,
        upto=args.upto)
