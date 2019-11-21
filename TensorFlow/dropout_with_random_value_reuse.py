# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrappers for primitive Neural Net (NN) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.variable_scope import  get_variable
from tensorflow.python.ops import random_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import variable_scope
import tensorflow as tf

#README
#========================================
# Adapted version of Dropput do be compatible with the PAL optimizer
# To reuse random values for more than one step session.run the toggle command switch_reuse_binary_tensor_op
# To use new random values switch the toggle back
#========================================

switch_reuse_binary_tensor_op=None
_is_reuse_binary_tensor_var=None

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn





def _get_noise_shape(x, noise_shape):
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return x.get_shape()
    #return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape


@tf_export(v1=["nn.dropout"])
@deprecation.deprecated_args(None, "Please use `rate` instead of `keep_prob`. "
                             "Rate should be set to `rate = 1 - keep_prob`.",
                             "keep_prob")
def dropout(x, keep_prob=None, noise_shape=None, seed=None, name=None,
            rate=None):  # pylint: disable=invalid-name
  """Computes dropout.
  For each element of `x`, with probability `rate`, outputs `0`, and otherwise
  scales up the input by `1 / (1-rate)`. The scaling is such that the expected
  sum is unchanged.
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.
  Args:
    x: A floating point tensor.
    keep_prob: (deprecated) A deprecated alias for `(1-rate)`.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed` for behavior.
    name: A name for this operation (optional).
    rate: A scalar `Tensor` with the same type as `x`. The probability that each
      element of `x` is discarded.
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating
      point tensor.
  """
  try:
    keep = 1. - keep_prob if keep_prob is not None else None
  except TypeError:
    raise ValueError("keep_prob must be a floating point number or Tensor "
                     "(got %r)" % keep_prob)

  rate = deprecation.deprecated_argument_lookup(
      "rate", rate,
      "keep_prob", keep)

  if rate is None:
    raise ValueError("You must provide a rate to dropout.")

  return dropout_v2(x, rate, noise_shape=noise_shape, seed=seed, name=name)


@tf_export("nn.dropout", v1=[])
def dropout_v2(x, rate, is_training, noise_shape=None, seed=None, name=None):  # pylint: disable=invalid-name
  """Computes dropout.
  With probability `rate`, drops elements of `x`. Input that are kept are
  scaled up by `1 / (1 - rate)`, otherwise outputs `0`.  The scaling is so that
  the expected sum is unchanged.
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.
  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
      for behavior.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
      point tensor.
  """
  with ops.name_scope(name, "dropout", [x]) as name:


    x = ops.convert_to_tensor(x, name="x")
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(rate, numbers.Real) and not (rate >= 0 and rate < 1):
      raise ValueError("rate must be a scalar tensor or a float in the "
                       "range [0, 1), got %g" % rate)

    # Early return if nothing needs to be dropped.
    if isinstance(rate, numbers.Real) and rate == 0:
      return x
    if context.executing_eagerly():
      if isinstance(rate, ops.EagerTensor):
        if rate.numpy() == 0:
          return x
    else:
      rate = ops.convert_to_tensor(
          rate, dtype=x.dtype, name="rate")
      rate.get_shape().assert_is_compatible_with(tensor_shape.scalar())

      # Do nothing if we know rate == 0
      if tensor_util.constant_value(rate) == 0:
        return x
    noise_shape = _get_noise_shape(x, noise_shape)


    global _is_reuse_binary_tensor_var
    global switch_reuse_binary_tensor_op
    if _is_reuse_binary_tensor_var is None:
      _is_reuse_binary_tensor_var= tf.get_variable("reuse_binary_tensor_var", initializer=tf.constant(False), trainable=False, dtype=tf.bool)
      switch_reuse_binary_tensor_op = tf.cond(_is_reuse_binary_tensor_var,
                                              lambda: _is_reuse_binary_tensor_var.assign(False),
                                              lambda: _is_reuse_binary_tensor_var.assign(True))

    binary_tensor_var = tf.get_variable("dropout_binary_tensor", noise_shape, initializer=tf.zeros_initializer(),trainable=False)
    # set reuse flag always to false after applying dropout


    def apply_dropout():
      keep_prob = 1 - rate
      # uniform [keep_prob, 1.0 + keep_prob)
      random_tensor = keep_prob
      random_tensor += random_ops.random_uniform(
        noise_shape, seed=seed, dtype=x.dtype)
      # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
      binary_tensor = math_ops.floor(random_tensor)

      # save binary tensor to variable
      assign_op = binary_tensor_var.assign(binary_tensor)
      with tf.control_dependencies([assign_op]):
        ret = math_ops.divide(x, keep_prob) * binary_tensor
        #ret = tf.Print(ret,["apply dropout and save", _is_reuse_binary_tensor_var])
      return ret


    def re_apply_dropout():
      keep_prob = 1 - rate
      ret = math_ops.divide(x, keep_prob) * binary_tensor_var
      #ret = tf.Print(ret, ["reapply dropout", _is_reuse_binary_tensor_var])
      return ret

    #drop_alt =
    # cond2 = lambda: tf.cond(_is_reuse_binary_tensor_var, lambda: re_apply_dropout(x,rate),lambda: apply_dropout(x,noise_shape,seed,rate))
    cond2 = lambda: tf.cond(_is_reuse_binary_tensor_var, re_apply_dropout, apply_dropout)
    ret = tf.cond(tf.logical_not(is_training), lambda: x, cond2)
    # TODO check whether dropout gets not executed during training

    #tf.equal(is_training, tf.constant(False)

    if not context.executing_eagerly():
      ret.set_shape(x.get_shape())
    return ret


  # """Apply drop connect."""
  # batch_size = tf.shape(inputs)[0]
  # shape = [batch_size, 1, 1, 1]
