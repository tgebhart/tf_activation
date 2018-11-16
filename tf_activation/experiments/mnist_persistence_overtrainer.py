from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data
from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer
from tf_activation.models.cff import CFF_Model
from tf_activation.models.fff import FFF_Model
from tf_activation.models.ccff import CCFF_Model

persistence_module = tf.load_op_library('/home/gebha095/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_train_persistence.so')

SEED = 1
STD = 0.1
TRAIN_SAMPLE_NUM = 55000
BATCH_SIZE = 128
P = 99
H = 0
EPOCHS = 51
OPTIMIZER = 'SGD'
LEARNING_RATE = 1e-4

def plot_diagram(diag, n, i):

    ax = plt.subplot()

    ax.scatter(diag[:,0], diag[:,1], s=25, c=(diag[:,0] - diag[:,1])**2, cmap=plt.cm.coolwarm, zorder=10)
    lims = [
        np.min(.9*diag[:,0]),  # min of both axes
        np.max(1.1*diag[:,1]),  # max of both axes
    ]

    # now plot both limits against eachother

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('Birth Time')
    plt.ylabel('Death Time')

    plt.savefig(os.path.join(n, 'diagram_' + str(i) + '.png'), dpi=1200,
                            format='png', bbox_inches='tight')

    plt.close()
    plt.clf()
    plt.cla()

def persistence_score(diag, stddev):
    return np.sum(np.square(diag[:,0] - diag[:,1]))/stddev

def avg_persistence_score(diag, stddev):
    return persistence_score(diag, stddev)/diag.shape[0]

def get_train_set(classes):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    if len(classes) == 0:
        return mnist.train.images, mnist.train.labels
    else:
        train_msk = np.where(np.isin(np.argmax(mnist.train.labels, axis=1), classes))
        return mnist.train.images[train_msk], mnist.train.labels[train_msk]

def get_test_set(classes):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    if len(classes) == 0:
        return mnist.test.images, mnist.test.labels
    else:
        test_msk = np.where(np.isin(np.argmax(mnist.test.labels, axis=1), classes))
        return mnist.test.images[test_msk], mnist.test.labels[test_msk]


def run(diagram_directory, model_directory=None, epochs=EPOCHS, p=P,
        train_sample_num=TRAIN_SAMPLE_NUM, batch_size=BATCH_SIZE, optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE, h=H, stddev=STD, classes=[]):

    # model = CFF_Model(W_conv1=[3,3,1,16], stddev=stddev)
    # model = CFF_Model(W_conv1=[5,5,1,32], stddev=stddev)
    # model = FFF_Model(stddev=stddev)
    model = CCFF_Model(W_conv2=[3, 3, 16, 32], stddev=stddev)

    train_string = model.get_trained_model_string('mnist-overtrain', optimizer, learning_rate, epochs, batch_size, stddev).replace('.ckpt', '')
    if len(classes) > 0:
            diagram_directory = os.path.join(diagram_directory, model.get_model_string(), ''.join(str(cls) for cls in classes), train_string)
    else:
        h_string = 'h' + str(h) + '_'
        diagram_directory = os.path.join(diagram_directory, model.get_model_string(), h_string + train_string)
    if not os.path.exists(diagram_directory):
        os.makedirs(diagram_directory)


    if model_directory is not None:
        if len(classes) > 0:
            model_directory = os.path.join(model_directory, model.get_model_string(), ''.join(str(cls) for cls in classes), train_string)
        else:
            model_directory = os.path.join(model_directory, model.get_model_string(), train_string)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allocator_type = 'BFC'
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

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=model.get_output_layer()))
        train_step = model.get_train_step(cross_entropy, optimizer=optimizer, lr=learning_rate)
        correct_prediction = tf.equal(tf.argmax(model.get_output_layer(), 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        resdf = []

        sess.run(tf.global_variables_initializer())

        trainX, trainY = get_train_set(classes)
        testX, testY = get_test_set(classes)

        for epoch in range(epochs):

            acc_idx = np.arange(1000)

            percentiles = persistence_module.layerwise_percentile(model.get_persistence_structure(),
                                                                    model.get_persistence_numbers(),
                                                                    [p,p,p])

            percentile = percentiles.eval(feed_dict={x: trainX[acc_idx[:1]], keep_prob:1.0})

            diagram_filename = os.path.join(diagram_directory, 'diagram_' + str(epoch) + '.csv')

            result = persistence_module.input_graph_persistence(model.get_persistence_structure(),
                                                                model.get_persistence_numbers(),
                                                                np.stack((percentile, percentile)),
                                                                h,
                                                                diagram_filename
                                                                )
            r = result.eval(feed_dict={x:trainX[acc_idx[:1]], keep_prob:1.0})

            diag = np.genfromtxt(diagram_filename, delimiter=',')

            per_score = avg_persistence_score(r, stddev)

            plot_diagram(diag, diagram_directory, epoch)

            train_accuracy = accuracy.eval(feed_dict={x: trainX[acc_idx], y_: trainY[acc_idx], keep_prob: 1.0})
            print("Epoch: %d, training accuracy %g, persistence score %g" % (epoch, train_accuracy, per_score))
            for i in range(trainX.shape[0]//batch_size):
                batchX = trainX[acc_idx]
                batchY = trainY[acc_idx]
                train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})

            test_accuracy = accuracy.eval(feed_dict={
                x: testX, y_: testY, keep_prob: 1.0})
            print('test accuracy %g' % test_accuracy)
            resdf.append({'epoch':epoch, 'persistence_score':per_score, 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy})

        resdf = pd.DataFrame(resdf)
        resdf.to_pickle(os.path.join(diagram_directory, 'resdf.pkl'))

        if model_directory is not None:
            save_path = saver.save(sess,model_directory)
            print("model saved in file: {}".format(save_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--diagram_directory",
                      required=True,
                      help="directory in which to save persistence diagram visualizations")
    parser.add_argument("-m","--model_directory",
                      required=False,
                      help="directory in which to save trained model")
    parser.add_argument("-p","--percentile",
                      type=float,
                      required=False,
                      default=P,
                      help="The percentile filtration of edge weights")
    parser.add_argument("-e","--epochs",
                      type=int,
                      required=False,
                      default=EPOCHS,
                      help="Number of epochs to train on")
    parser.add_argument("-hd","--homology_dimension",
                      type=int,
                      required=False,
                      default=H,
                      help="Homology on which dimension")
    parser.add_argument("-t","--train_samples",
                      type=int,
                      required=False,
                      default=TRAIN_SAMPLE_NUM,
                      help="Number of training samples per epoch")
    parser.add_argument("-bs","--batch_size",
                      required=False,
                      type=int,
                      default=BATCH_SIZE,
                      help="The batch size for training")
    parser.add_argument("-lr","--learning_rate",
                      type=float,
                      required=False,
                      default=LEARNING_RATE,
                      help="Learning rate for optimizer for training")
    parser.add_argument("-opt","--optimizer",
                      required=False,
                      default=OPTIMIZER,
                      help="choice of backprop optimizer")
    parser.add_argument("-std","--stddev",
                      required=False,
                      type=float,
                      default=STD,
                      help="standard deviation of weight initialization normal")
    parser.add_argument('-c','--classes',
                        required=False,
                        nargs='+',
                        default=[],
                        help='List of classes to train on (0-9)')


    args = parser.parse_args()
    classes = list(map(lambda x: int(x), args.classes))
    run(args.diagram_directory, model_directory=args.model_directory, epochs=args.epochs, p=args.percentile,
        train_sample_num=args.train_samples, batch_size=args.batch_size,
        optimizer=args.optimizer, learning_rate=args.learning_rate, h=args.homology_dimension,
        stddev=args.stddev, classes=classes)
