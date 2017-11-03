from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.gen_nn_ops import *

import tensorflow as tf

local_response_normalization = gen_nn_ops.lrn


def persistent_dropout(x, kelsea, keepout=False, seed=None, name=None):  # pylint: disable=invalid-name
  """Computes dropout.
  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.
  Args:
    x: A tensor.
    kelsea: A list with the indices of x that are to be dropped
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.

  """
  with ops.name_scope(name, "persistent_dropout", [x]) as name:

    indices = tf.Variable(initial_value=kelsea)

    updates = tf.zeros(indices.get_shape().as_list()[1])
    mul = tf.scatter_nd(indices, updates, [x.get_shape().as_list()[1]])

    if keepout:
        ret = x * mul
    else:
        ret = x * (tf.ones(mul.get_shape()) - mul)
    return ret


# def persistent_dropout(x, kelsea, keepout=False, seed=None, name=None):  # pylint: disable=invalid-name
#   """Computes dropout.
#   With probability `keep_prob`, outputs the input element scaled up by
#   `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
#   sum is unchanged.
#   By default, each element is kept or dropped independently.  If `noise_shape`
#   is specified, it must be
#   [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
#   to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
#   will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
#   and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
#   kept independently and each row and column will be kept or not kept together.
#   Args:
#     x: A tensor.
#     kelsea: A list with the indices of x that are to be dropped
#     seed: A Python integer. Used to create random seeds. See
#       @{tf.set_random_seed}
#       for behavior.
#     name: A name for this operation (optional).
#   Returns:
#     A Tensor of the same shape of `x`.
#
#   """
#   with ops.name_scope(name, "persistent_dropout", [x]) as name:
#
#     # x_ret = tf.get_variable("persistent_dropout_out", initializer=x,validate_shape=False)
#     input_var = tf.Variable(x, validate_shape=False)
#
#     if keepout:
#         ret = tf.zeros(array_ops.shape(x))
#         # ret = tf.zeros(array_ops.shape(x), dtype=x.dtype)
#         for idx in kelsea:
#             ret[idx] = x[idx[:]]
#     else:
#         k = kelsea.eval()
#         print(x.get_shape())
#         ret = np.zeros(shape=x.get_shape().as_list()[1])
#         print(ret.shape)
#         for idx in k:
#             ret[idx] = 1.0
#
#     # tf.scatter_update(x_ret, kelsea, np.zeros(shape=x.get_shape().as_list()[1:],  dtype=np.float32))
#     # x_ret = ops.convert_to_tensor(ret, name="persistent_dropout_out")
#     mul = x * ops.convert_to_tensor(ret)
#     return mul
