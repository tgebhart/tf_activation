from tf_activation.models.cff import CFF_Model
from tf_activation.models.fff import FFF_Model
from tf_activation.models.ccff import CCFF_Model
import tf_activation.functions.plotnn as pltnn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

persistence_module = tf.load_op_library('/home/gebha095/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')

from time import time
import argparse
import os
from functools import wraps
import errno
import os
import signal

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

P = 99
UPTO = 1000

def run(output_location, model_directory, trained_model, p=P, classes=[],
        upto=UPTO, test_set=False):

    model = CFF_Model(W_conv1=[5,5,1,32])

    model.infer_graph_parameters_from_filename(model_directory)
    training_parameters = model.infer_training_parameters_from_filename(trained_model)

    print(training_parameters)

    model_string = model_directory[model_directory.rfind('/')+1:]
    train_string = trained_model[:trained_model.rfind('/')]


    p_string = 'p{}_'.format(p).replace('.', 'p')
    filtration_directory = os.path.join(output_location, model_string, p_string + train_string)
    if not os.path.exists(filtration_directory):
        os.makedirs(filtration_directory)


    config = tf.ConfigProto()
    config.log_device_placement = False

    with tf.device('/cpu:0'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        net = model.build_graph(x)
        if not model.implements_dropout:
            keep_prob = tf.placeholder(tf.float32)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=model.get_output_layer()))
        train_step = model.get_train_step(cross_entropy, optimizer=training_parameters['optimizer'], lr=training_parameters['learning_rate'])
        correct_prediction = tf.equal(tf.argmax(model.get_output_layer(), 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        saver.restore(sess, os.path.join(model_directory, trained_model))

        filtration_suffix = ''
        if test_set:
            X = mnist.test.images
            Y = mnist.test.labels
            filtration_suffix = '_test.csv'
        else:
            X = mnist.train.images
            Y = mnist.train.labels
            filtration_suffix = '_train.csv'

        for i in range(upto):
            print("i: {}".format(i))

            correct_label = np.argmax(Y[i])

            # skip this iteration if we come across a class we are not interested in
            if len(classes) > 0 and correct_label not in classes:
                continue

            filtration_info_loc = os.path.join(filtration_directory, str(correct_label))
            if not os.path.exists(filtration_info_loc):
                os.makedirs(filtration_info_loc)
            filtration_filename = os.path.join(filtration_info_loc, str(i)+filtration_suffix)
            # filtration_extra_filename = os.path.join(filtration_info_loc, str(i) + '.csv')

            test_inputs = np.stack((X[i], X[i]))
            test_labels = np.stack((Y[i], Y[i]))


            percentiles = persistence_module.layerwise_percentile(model.get_persistence_structure(),
                                                                    model.get_persistence_numbers(),
                                                                    [p,p,p])

            ps1 = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})
            # ps2 = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})

            result = persistence_module.input_graph_filtration(model.get_persistence_structure(),
                                                                model.get_persistence_numbers(),
                                                                np.stack((ps1, ps1)),
                                                                "_")


            r = result.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})

            np.savetxt(filtration_filename, r, delimiter=',')



def get_train_set(classes):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    if len(classes) == 0:
        return mnist.train.images, mnist.train.labels
    else:
        train_msk = np.where(np.isin(np.argmax(mnist.train.labels, axis=1), classes))
        return mnist.train.images[train_msk], mnist.train.labels[train_msk]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--output_location",
                      required=True,
                      help="directory in which to store subgraph representations")
    parser.add_argument("-m","--model_directory",
                      required=True,
                      help="directory location of trained model to produce graph")
    parser.add_argument("-t","--trained_model",
                      required=True,
                      help="name of trained model checkpoint")
    parser.add_argument("-p","--percentile",
                      type=float,
                      required=False,
                      default=P,
                      help="The percentile filtration of edge weights")
    parser.add_argument('-c','--classes',
                        required=False,
                        nargs='+',
                        default=[],
                        help='List of classes to train on (0-9)')
    parser.add_argument('-ut','--upto',
                        type=int,
                        required=False,
                        default=UPTO,
                        help="Number examples to create persistent subgraphs from")
    parser.add_argument('-te','--test',
                        required=False,
                        action='store_true',
                        help="Whether to run on test set")


    args = parser.parse_args()
    classes = list(map(lambda x: int(x), args.classes))
    run(args.output_location, args.model_directory, args.trained_model, p=args.percentile,
        classes=classes, upto=args.upto, test_set=args.test)
