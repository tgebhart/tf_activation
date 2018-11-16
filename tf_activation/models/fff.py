from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import pickle

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

from tf_activation import DeepGraph
from tf_activation import ConvolutionLayer
from tf_activation.models.base import Base_Model

class FFF_Model(Base_Model):

    def __init__(self, activation='relu',
                input=[-1, 28, 28, 1],
                W_fc1 = 1024,
                W_fc2 = 1024,
                W_fc3 = 10,
                seed = None,
                stddev = 0.1):

        self.input = input
        self.W_fc1 = W_fc1
        self.W_fc2 = W_fc2
        self.W_fc3 = W_fc3
        super().__init__(activation=activation, seed=seed, stddev=stddev)

        self.graph = {'input':None,
                          'W_fc1':None,
                          'h_fc1':None,
                          'W_fc2':None,
                          'h_fc2':None,
                          'W_fc3':None,
                          'y':None}

    def get_persistence_structure(self):
        return [
        self.graph['input'],
        self.graph['W_fc1'],
        self.graph['h_fc1'],
        self.graph['W_fc2'],
        self.graph['h_fc2'],
        self.graph['W_fc3'],
        self.graph['y']
        ]

    def get_persistence_numbers(self):
        return [0, 2, 4, 1, 4, 1, 4]

    def get_model_string(self):
        return 'fff_wfc1{}_wfc2{}_wfc3{}_seed{}_{}'.format(str(self.W_fc1),
                                                    str(self.W_fc2),
                                                    str(self.W_fc3),
                                                    str(self.seed), self.activation)

    def get_trained_model_string(self, dataset, optimizer, learning_rate, epochs, batch_size, stddev):
        return '{}_{}{:.1e}_{}epochs_{}batch_{}std'.format(dataset, optimizer,
                learning_rate, str(epochs), str(batch_size), str(stddev)).replace('.','p')+'.ckpt'

    def get_restore_string(self, dataset, optimizer, learning_rate, epochs, batch_size, stddev):
        return os.path.join(self.get_model_string(),
                            self.get_trained_model_string(dataset, optimizer, learning_rate, epochs, batch_size, stddev))

    def get_output_layer(self):
        return self.graph['y']

    def build_graph(self, x):
       """builds the graph for a deep net """

       with tf.device('/cpu:0'):
           # Reshape to use within a convolutional neural net.
           # Last dimension is for "features" - there is only one here, since images are
           # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
           self.graph['input'] = tf.reshape(x, self.input, name='input')

           flattening = self.input[1]*self.input[2]
           reshaped = tf.reshape(self.graph['input'], [-1, flattening])
           self.graph['W_fc1'] = self.weight_variable([flattening, self.W_fc1])
           self.graph['h_fc1'] = self.activate(tf.matmul(reshaped, self.graph['W_fc1']))

           # Map the 1024 features to 10 classes, one for each digit
           self.graph['W_fc2'] = self.weight_variable([self.W_fc1, self.W_fc2])
           self.graph['h_fc2'] = tf.matmul(self.graph['h_fc1'], self.graph['W_fc2'])


           self.graph['W_fc3'] = self.weight_variable([self.W_fc2, self.W_fc3])
           self.graph['y'] = tf.matmul(self.graph['h_fc2'], self.graph['W_fc3'])

           return self.graph



if __name__ == "__main__":
    print('FFF Model')
