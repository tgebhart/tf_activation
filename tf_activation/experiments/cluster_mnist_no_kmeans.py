from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
# from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

num_steps = 54999

centers = np.zeros(shape=(mnist.train.images[0].shape[0], mnist.train.labels[0].shape[0]))

for i in range(num_steps):

    center = np.argmax(mnist.train.labels[i])

    centers[:,center] += mnist.train.images[i]


centers = np.divide(centers, num_steps)

np.save('../../logdir/models/average_mnist', centers, allow_pickle=True)
