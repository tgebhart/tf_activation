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
H = 0
UPTO = 1000
DP = 0.1
GRAPH_TYPE = 'circular'

def run(output_location, model_directory, trained_model, h=H, p=P, classes=[],
        upto=UPTO, dp=DP, test_set=False, graph_type=GRAPH_TYPE):

    model = CFF_Model(W_conv1=[5,5,1,32])

    model.infer_graph_parameters_from_filename(model_directory)
    training_parameters = model.infer_training_parameters_from_filename(trained_model)

    print(training_parameters)

    model_string = model_directory[model_directory.rfind('/')+1:]
    train_string = trained_model[:trained_model.rfind('/')]

    if len(classes) > 0:
        h_string = 'h' + str(h) + '_'
        p_string = 'p{}_'.format(p).replace('.', 'p')
        graph_directory = os.path.join(output_location, model_string, ''.join(str(cls) for cls in classes), h_string + p_string + train_string)
    else:
        h_string = 'h' + str(h) + '_'
        p_string = 'p{}_'.format(p).replace('.', 'p')
        graph_directory = os.path.join(output_location, model_string, h_string + p_string + train_string)
    if not os.path.exists(graph_directory):
        os.makedirs(graph_directory)


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

        graph_suffix = ''
        if test_set:
            X = mnist.test.images
            Y = mnist.test.labels
            graph_suffix = '_test.graphml'
        else:
            X = mnist.train.images
            Y = mnist.train.labels
            graph_suffix = '_train.graphml'

        for i in range(upto):
            print("i: {}".format(i))

            correct_label = np.argmax(Y[i])

            # skip this iteration if we come across a class we are not interested in
            if len(classes) > 0 and correct_label not in classes:
                continue

            graph_info_loc = os.path.join(graph_directory, str(correct_label))
            if not os.path.exists(graph_info_loc):
                os.makedirs(graph_info_loc)
            graphml_filename = os.path.join(graph_info_loc, str(i)+graph_suffix)


            test_inputs = np.stack((X[i], X[i]))
            test_labels = np.stack((Y[i], Y[i]))


            percentiles = persistence_module.layerwise_percentile(model.get_persistence_structure(),
                                                                    model.get_persistence_numbers(),
                                                                    [p,p,p])

            ps1 = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})
            ps2 = percentiles.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0})

            result = persistence_module.persistent_sub_graph(model.get_persistence_structure(),
                                                                model.get_persistence_numbers(),
                                                                np.stack((ps1,ps2)),
                                                                dp,
                                                                h,
                                                                graphml_filename)


            print(result.eval(feed_dict={x: test_inputs[0:1], keep_prob:1.0}))


            g = nx.read_graphml(graphml_filename)

            if graph_type == 'circular':
                nx.draw_circular(g, node_size=5)
                plt.savefig(os.path.join(graph_info_loc, '{}_circular_'.format(dp).replace('.', 'p') + str(i) + graph_suffix.replace('.graphml', '.png')),
                            format='png', dpi=1000)
                plt.close()
            if graph_type == 'spring':
                nx.draw_spring(g, node_size=5)
                plt.savefig(os.path.join(graph_info_loc, '{}_spring_'.format(dp).replace('.', 'p') + str(i) + graph_suffix.replace('.graphml', '.png')),
                            format='png', dpi=1000)
            if graph_type == 'spectral':
                nx.draw_spectral(g, node_size=5)
                plt.savefig(os.path.join(graph_info_loc, '{}_spectral_'.format(dp).replace('.', 'p') + str(i) + graph_suffix.replace('.graphml', '.png')),
                            format='png', dpi=1000)
                plt.close()
            # if graph_type == 'custom':
            #     network = pltnn.DrawNN(model.get_layerwise_dimensions())
            #     network.draw()


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
    parser.add_argument("-dp","--diagram_percentile",
                      type=float,
                      required=False,
                      default=DP,
                      help="The percentile of points in the diagram to take in units of lifetime")
    parser.add_argument("-hd","--homology_dimension",
                      type=int,
                      required=False,
                      default=H,
                      help="Homology on which dimension")
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
    parser.add_argument("-gt","--graph_type",
                      required=True,
                      help="type of graph to draw")


    args = parser.parse_args()
    classes = list(map(lambda x: int(x), args.classes))
    run(args.output_location, args.model_directory, args.trained_model, p=args.percentile,
        h=args.homology_dimension, classes=classes, upto=args.upto, dp=DP, test_set=args.test,
        graph_type=args.graph_type)
