# Copyright (c) 2021, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of Google Inc. nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Survival Network."""

import tensorflow as tf


def cox_partial_likelihood(event_times, censored, preds):
  """Returns negative log of cox partial liklihood.

  This implementation uses Breslow's approximation for handling ties. For
  details on Breslow's method, see page 144 in
  https://www4.stat.ncsu.edu/~dzhang2/st745/chap7.pdf. Note that we calculate
  the loss with respect to the negative of `preds` such that preds are
  positively correlated with event times.

  Args:
    event_times: ground-truth event times. Tensor of shape [batch_size].
    censored: mask indicating whether the example is censored. Tensor of shape
      [batch_size].
    preds: predicted event times. Tensor of shape [batch_size].

  """
  mask = get_risk_set(event_times, ignore_ties=False)
  preds = shift_preds(preds)
  loss = preds - logsumexp_masked(tile_rows(preds), mask)
  observed = tf.cast(tf.logical_not(censored), loss.dtype)
  loss = tf.reduce_sum(loss * observed)
  loss = tf.math.divide_no_nan(loss, tf.reduce_sum(observed))
  loss = -loss  # we minimize the negative liklihood
  return loss


def get_risk_set(event_times, ignore_ties=False):
  """Returns a matrix where row i indicates the risk set for example i.

  If ignore_ties=True:
    m[i, j] == 1 iff j > i
  If ignore_ties=False:
    m[i, j] == 1 iff j >= i

  Args:
    event_times: 1D tenseor of event times
    ignore_ties: if False, comparable pairs can have tied event times.
  """
  m1 = tile_rows(event_times)
  m2 = tile_columns(event_times)
  if ignore_ties:
    return tf.greater(m1, m2)
  else:
    return tf.greater_equal(m1, m2)


def tile_rows(a):
  """Returns a matrix where each row is equal to `a`.

  Example:
  a =  [2, 1, 3]

  m = [[2, 1, 3],
       [2, 1, 3],
       [2, 1, 3]]

  Args:
    a: 1D tensor.
  """
  n = tf.shape(a)[0]
  return tf.tile(tf.expand_dims(a, axis=0), (n, 1))


def tile_columns(a):
  """Returns a matrix where each column is equal to `a`.

  Example:
  a =  [2, 1, 3]

  m = [[2, 2, 2],
       [1, 1, 1],
       [3, 3, 3]]

  Args:
    a: 1D tensor.
  """
  return tf.transpose(tile_rows(a))


def logsumexp_masked(a, mask):
  """Returns row-wise masked log sum exp of a.

  Uses the following trick for numeric stability:
  log(sum(exp(x))) == log(sum(exp(x - max(x)))) + max(x)

  Args:
    a: 2D tensor.
    mask: 2D tensor.
  """
  mask = tf.cast(mask, a.dtype)
  a_max = tf.math.reduce_max(a * mask, axis=1, keepdims=True)
  a = a - a_max
  a_exp = tf.math.exp(a)
  a_sum_exp = tf.math.reduce_sum(a_exp * mask, axis=1, keepdims=True)
  return tf.squeeze(tf.math.log(a_sum_exp) + a_max)


def shift_preds(preds):
  """Returns uniformly shift preds so minimum is at 0 to avoid underflow.

  Args:
    preds: Tensor of shape [batch_size].
  """
  preds_min = tf.reduce_min(preds)
  shift = tf.where(preds_min < 0, -preds_min, 0)
  return preds + shift


def keras_cox_partial_likelihood(y_true, y_pred):
  """Keras friendly wrapper for cox_partial_likelihood."""
  event_times = tf.squeeze(tf.gather(y_true, [0], axis=1))
  censored = tf.cast(tf.squeeze(tf.gather(y_true, [1], axis=1)), tf.bool)
  preds = tf.squeeze(y_pred, axis=1)
  return cox_partial_likelihood(event_times, censored, preds)
