from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 1e-4
training_steps = 10000
batch_size = 128
display_step = 200
P = 99

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

persistence_module = tf.load_op_library('/home/tgebhart/python/tensorflow/bazel-bin/tensorflow/core/user_ops/nn_graph_persistence.so')

def plot_diagram(diag, n, i):

    ax = plt.subplot()

    ax.scatter(diag[:,0], diag[:,1], s=25, c=(diag[:,0] - diag[:,1])**2, cmap=plt.cm.coolwarm, zorder=10)
    lims = [
        np.min([0]),  # min of both axes
        np.max(diag[:,0]),  # max of both axes
    ]

    # now plot both limits against eachother

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('Birth Time')
    plt.ylabel('Death Time')

    plt.savefig(os.path.join(n, 'diagram_' + str(i) + '.svg'), dpi=1200,
                            format='svg', bbox_inches='tight')

    plt.close()
    plt.clf()
    plt.cla()

def RNN(x, weights, biases):
    ret = {}
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    ret['input'] = x

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    ret['recurrents'] = outputs
    ret['lstm'] = outputs[-1]

    ret['W'] = weights['out']
    # Linear activation, using rnn inner loop last output
    logits = tf.matmul(ret['lstm'], weights['out']) + biases['out']
    ret['logits'] = logits
    return logits, ret

logits, net = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:

            percentiles = persistence_module.layerwise_percentile([net['input'],
                                                                net['recurrents'],
                                                                net['lstm'],
                                                                net['lstm'],
                                                                net['W'],
                                                                net['logits']],
                                                                [0,1,6,6,1,4],
                                                                [P,P])

            ps1 = percentiles.eval(feed_dict={X: batch_x[:1]})
            print('Computed Percentile:', ps1)

            diagram_filename = '../../logdir/data/diagram_rnn.csv'

            result = persistence_module.input_graph_persistence([net['input'],
                                                                net['recurrents'],
                                                                net['lstm'],
                                                                net['lstm'],
                                                                net['W'],
                                                                net['logits']],
                                                                [0,1,6,6,1,4],
                                                                np.stack((ps1, ps1)),
                                                                0,
                                                                diagram_filename
                                                                )
            r = result.eval(feed_dict={X: batch_x[:1]})

            diag = np.genfromtxt(diagram_filename, delimiter=',')

            plot_diagram(diag, '../../logdir/data', 0)

            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 1000
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
