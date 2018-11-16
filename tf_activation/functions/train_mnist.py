import argparse
import sys
import os
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data
from tf_activation.models.cff import CFF_Model
from tf_activation.models.fff import FFF_Model
from tf_activation.models.ccff import CCFF_Model

SEED = 1
STD = 0.1
TRAIN_SAMPLE_NUM = 55000
BATCH_SIZE = 48
P = 99
H = 0
EPOCHS = 51
OPTIMIZER = 'SGD'
LEARNING_RATE = 1e-4
RANDOM_INITIALIZATION = 'normal'
ACTIVATION = 'relu'

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


def run(model_directory=None, epochs=EPOCHS,
        train_sample_num=TRAIN_SAMPLE_NUM, batch_size=BATCH_SIZE, optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE, stddev=STD, classes=[], activation=ACTIVATION,
        initialization=RANDOM_INITIALIZATION, seed=SEED):

    # model = CFF_Model(W_conv1=[3,3,1,16], stddev=stddev, )
    model = CFF_Model(W_conv1=[5,5,1,32], stddev=stddev, activation=activation, initializer=initialization, seed=seed)
    # model = FFF_Model(stddev=stddev)
    # model = CCFF_Model(W_conv2=[3, 3, 16, 32], stddev=stddev)

    train_string = model.get_trained_model_string('mnist', optimizer, learning_rate, epochs, batch_size).replace('.ckpt', '')
    if model_directory is not None:
        if len(classes) > 0:
            model_directory = os.path.join(model_directory, model.get_model_string(), ''.join(str(cls) for cls in classes), train_string)
        else:
            model_directory = os.path.join(model_directory, model.get_model_string(), train_string)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allocator_type = 'BFC'
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
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(y_, dtype=tf.int32), logits=logits+np.finfo(float).eps))
        train_step = model.get_train_step(cross_entropy, optimizer, lr=learning_rate)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    with tf.device('/cpu:0'):

        saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        resdf = []

        sess.run(tf.global_variables_initializer())

        trainX, trainY = get_train_set(classes)
        testX, testY = get_test_set(classes)

        for epoch in range(epochs):

            acc_idx = np.arange(1000)

            p = np.random.permutation(trainX.shape[0])
            trainX = trainX[p]
            trainY = trainY[p]

            train_accuracy = accuracy.eval(feed_dict={x: trainX[acc_idx], y_: trainY[acc_idx], keep_prob: 1.0})
            print("Epoch: %d, training accuracy %g" % (epoch, train_accuracy))
            for i in range(trainX.shape[0]//batch_size):
                batchX = trainX[i*batch_size:i*batch_size+batch_size]
                batchY = trainY[i*batch_size:i*batch_size+batch_size]
                train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})

            test_accuracy = accuracy.eval(feed_dict={x: testX[acc_idx], y_: testY[acc_idx], keep_prob: 1.0})
            print('test accuracy %g' % test_accuracy)

            train_loss = cross_entropy.eval(feed_dict={x: trainX[acc_idx], y_: trainY[acc_idx], keep_prob:1.0})
            test_loss = cross_entropy.eval(feed_dict={x: testX[acc_idx], y_: testY[acc_idx], keep_prob:1.0})
            print('train loss: {} test loss: {}'.format(train_loss, test_loss))

            resdf.append({'epoch':epoch, 'train_accuracy':train_accuracy, 'test_accuracy':test_accuracy, 'train_loss':train_loss, 'test_loss':test_loss})

        resdf = pd.DataFrame(resdf)

        # resdf.to_pickle(os.path.join(diagram_directory, 'resdf.pkl'))

        if model_directory is not None:
            save_path = saver.save(sess,os.path.join(model_directory, 'model.ckpt'))
            print("model saved in file: {}".format(save_path))

        fig, axp = plt.subplots()
        axp.set_xlabel('epoch')
        axp.set_ylabel('accuracy')
        axp.plot(resdf['epoch'], resdf['test_accuracy'], 'bo-', linewidth=1, markersize=6)

        axa = axp.twinx()
        axa.set_ylabel('loss')
        axa.plot(df['epoch'], df['test_loss'], 'ro-', linewidth=1, markersize=6)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-m","--model_directory",
                      required=False,
                      help="directory in which to save trained model")
    parser.add_argument("-e","--epochs",
                      type=int,
                      required=False,
                      default=EPOCHS,
                      help="Number of epochs to train on")
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
                      help="standard deviation of weight initialization normal or range of uniform")
    parser.add_argument("-ri","--random_initialization",
                      required=False,
                      default=RANDOM_INITIALIZATION,
                      help="random initializer bias to use")
    parser.add_argument("-a","--activation",
                      required=False,
                      default=ACTIVATION,
                      help="activation function to use")
    parser.add_argument('-c','--classes',
                        required=False,
                        nargs='+',
                        default=[],
                        help='List of classes to train on (0-9)')
    parser.add_argument("-s","--seed",
                      required=False,
                      type=int,
                      default=SEED,
                      help="Random seed")


    args = parser.parse_args()
    classes = list(map(lambda x: int(x), args.classes))
    run(model_directory=args.model_directory, epochs=args.epochs,
        train_sample_num=args.train_samples, batch_size=args.batch_size,
        optimizer=args.optimizer, learning_rate=args.learning_rate,
        stddev=args.stddev, classes=classes, activation=args.activation,
        initialization=args.random_initialization,
        seed=args.seed)
