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
"""Train network on synthetic data."""

import lifelines
import numpy as np
import tensorflow as tf

import loss
import network

NUM_EXAMPLES = 64
SEQUENCE_LENGTH = 2
PATCH_SIZE = 128
NUM_EPOCHS = 32


def main():
  # Set up synthetic data
  rs = np.random.RandomState(0)
  shape = (NUM_EXAMPLES, SEQUENCE_LENGTH, PATCH_SIZE, PATCH_SIZE, 3)
  images = rs.rand(*shape)
  event_times = rs.rand(NUM_EXAMPLES)
  censored = rs.rand(NUM_EXAMPLES) > 0.75
  y_true = np.stack([event_times, censored], axis=1)

  # Build network
  model = network.build_network(images.shape[1:])
  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.01),
      loss=loss.keras_cox_partial_likelihood,
      metrics=[])

  # Compute baseline c-index
  y_pred_baseline = -model.predict(images)[:, 0]
  c_index_init = lifelines.utils.concordance_index(
      event_times, y_pred_baseline, event_observed=~censored)
  print(f'Initial C-index: {c_index_init}')

  # Train model
  model.fit(images, y_true, epochs=NUM_EPOCHS)

  # Compute final c-index
  y_pred_final = -model.predict(images)[:, 0]
  c_index_final = lifelines.utils.concordance_index(
      event_times, y_pred_final, event_observed=~censored)
  print(f'Initial C-index: {c_index_init}')
  print(f'Final C-index:   {c_index_final}')

  # Assert that c-index increased by at least 10 points
  assert c_index_final > (c_index_init + 0.1)


if __name__ == '__main__':
  main()
