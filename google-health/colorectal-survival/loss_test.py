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
"""Tests for loss.py."""

import math
import numpy as np
import tensorflow as tf
from colorectal_survival import loss


class LossTest(tf.test.TestCase):

  def test_cox_partial_likelihood(self):
    with self.test_session():
      preds = tf.constant([6, 5, 4, 1], dtype=tf.float32)
      event_times = tf.constant([4, 5, 3, 2], dtype=tf.int32)
      censored = tf.constant([True, False, False, False], dtype=tf.bool)

      loss_1 = math.exp(5) / math.exp(5)
      loss_2 = math.exp(4) / (math.exp(6) + math.exp(5) + math.exp(4))
      loss_3 = math.exp(1) / (
          math.exp(6) + math.exp(5) + math.exp(4) + math.exp(1))
      expected = -(math.log(loss_1) + math.log(loss_2) + math.log(loss_3)) / 3
      actual = loss.cox_partial_likelihood(event_times, censored, preds)
      self.assertAllClose(expected, actual)

  def test_logsumexp_masked(self):
    with self.test_session():
      exp_a = tf.constant([[.3, .5, .2], [.1, .0, .9], [.2, .1, .7]])
      a = tf.math.log(exp_a)
      m = tf.constant([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
      expected = np.log([1.0, 0.1, 0.2])
      actual = loss.logsumexp_masked(a, m)
      self.assertAllClose(expected, actual)


if __name__ == '__main__':
  tf.test.main()
