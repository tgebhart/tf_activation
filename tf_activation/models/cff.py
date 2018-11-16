from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
from parse import *
import sys
import os
import pickle

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer
from tf_activation.models.base import Base_Model

class CFF_Model(Base_Model):

    def __init__(self, activation='relu',
                input=[-1, 28, 28, 1],
                W_conv1=[5, 5, 1, 32],
                W_fc1 = 1024,
                W_fc2 = 10,
                strides=[1, 1, 1, 1],
                seed = None,
                initializer = 'normal',
                stddev = 0.1):

        self.input = input
        self.W_conv1 = W_conv1
        self.W_fc1 = W_fc1
        self.W_fc2 = W_fc2
        super().__init__(activation=activation, strides=strides, seed=seed, stddev=stddev, initializer=initializer)

        self.graph = {'input':None,
                          'W_conv1':None,
                          'h_conv1':None,
                          'W_fc1':None,
                          'h_fc1':None,
                          'W_fc2':None,
                          'y':None}

        self.model_string = 'cff_wc1{}-{}-{}_wfc1{}_seed{}_{}std{}_{}'

    def get_persistence_structure(self, with_input=True):
        if with_input:
            return [
            self.graph['input'],
            self.graph['W_conv1'],
            self.graph['h_conv1'],
            self.graph['h_conv1'],
            self.graph['W_fc1'],
            self.graph['h_fc1'],
            self.graph['h_fc1'],
            self.graph['W_fc2'],
            self.graph['y']
            ]
        else:
            return [
            self.graph['W_conv1'],
            self.graph['h_conv1'],
            self.graph['h_conv1'],
            self.graph['W_fc1'],
            self.graph['h_fc1'],
            self.graph['h_fc1'],
            self.graph['W_fc2'],
            self.graph['y']
            ]

    def get_layerwise_dimensions(self):
        i = self.input[1] * self.input[2]
        wc1 = self.W_conv1[0] * self.W_conv1[1] * self.W_conv1[3]
        fc1 = self.W_fc1
        wc2 = self.W_fc2
        return [i,wc1,fc1,wc2]

    def get_persistence_numbers(self, with_input=True):
        if with_input:
            return [0, 1, 2, 2, 1, 4, 4, 1, 4]
        else:
            return [1, 2, 2, 1, 4, 4, 1, 4]

    def get_model_string(self):
        return self.model_string.format(str(self.W_conv1[0]),
                                        str(self.W_conv1[1]),
                                        str(self.W_conv1[3]),
                                        str(self.W_fc1),
                                        str(self.seed),
                                        self.initializer,
                                        str(self.stddev),
                                        self.activation).replace('.', 'p')

    def get_restore_string(self, dataset, optimizer, learning_rate, epochs, batch_size):
        return os.path.join(self.get_model_string(),
                            self.get_trained_model_string(dataset, optimizer, learning_rate, epochs, batch_size))

    def infer_graph_parameters_from_filename(self, s):
        if s.find('/') != -1:
            s = s[s.rfind('/')+1:]
        if s[:3] != 'cff':
            raise ValueError('Incorrect file string for this class of model; cannot infer parameters.')


        p = parse(self.model_string, s)

        self.W_conv1 = [int(p[0]), int(p[1]), 1, int(p[2])]
        self.W_fc1 = int(p[3])
        self.seed = ast.literal_eval(p[4])
        self.initializer = p[5]
        self.stddev = float(p[6].replace('p','.'))
        self.activation = p[7]


    def get_output_layer(self):
        return self.graph['y']

    def build_graph(self, x):
       """builds the graph for a deep net """

       with tf.device('/cpu:0'):
           # Reshape to use within a convolutional neural net.
           # Last dimension is for "features" - there is only one here, since images are
           # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
           self.graph['input'] = tf.reshape(x, self.input, name='input')

           # First convolutional layer - maps one grayscale image to feature maps.
           self.graph['W_conv1'] = self.weight_variable(self.W_conv1)
           self.graph['h_conv1'] = self.activate(self.conv2d(self.graph['input'], self.graph['W_conv1']))

           # tf.summary.scalar('W_conv1', ret['W_conv1'])

           flattening = self.input[1]*self.input[2]*self.W_conv1[3]
           reshaped = tf.reshape(self.graph['h_conv1'], [-1, flattening])

           self.graph['W_fc1'] = self.weight_variable([flattening, self.W_fc1])
           self.graph['h_fc1'] = self.activate(tf.matmul(reshaped, self.graph['W_fc1']))

           # Map the 1024 features to 10 classes, one for each digit
           self.graph['W_fc2'] = self.weight_variable([self.W_fc1, self.W_fc2])
           self.graph['y'] = tf.matmul(self.graph['h_fc1'], self.graph['W_fc2'])

           return self.graph



if __name__ == "__main__":
    print('CFF Model')
