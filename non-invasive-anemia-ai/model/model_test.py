# Copyright (c) 2026, Google Inc.
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

"""Unit tests for model module."""

from absl.testing import absltest
import torch
from model import config
from model import model


class ResNetMultimodalRegressorTest(absltest.TestCase):

  def test_same_pad_2d(self):
    """Tests the same padding function for 2D input."""
    x = torch.randn(2, 3, 256, 256)
    padded = model.same_pad_2d(x, kernel_size=7, stride=2)
    self.assertGreaterEqual(padded.shape[2], 256)
    self.assertGreaterEqual(padded.shape[3], 256)

  def test_same_pad_max_pool(self):
    """Tests the same padding max pooling function."""
    pooler = model.SamePadMaxPool()
    x = torch.randn(2, 64, 128, 128)
    pooled = pooler(x)
    self.assertEqual(pooled.shape, (2, 64, 64, 64))

  def test_forward_pass_returns_correct_shape(self):
    """Tests the forward pass of the ResNetMultimodalRegressor."""
    model_config = config.ResNetMultimodalConfig(
        num_channels=3,
        embedding_size=128,
        hidden_sizes=[128, 256],
        depths=[1, 1],
        layer_type='basic',
        metadata_size=3,
    )
    regressor = model.ResNetMultimodalRegressor(model_config)
    batch_size = 2
    image_input = torch.randn(batch_size, 3, 256, 256)
    metadata_input = torch.randn(batch_size, 3)
    output = regressor(image_input, metadata_input)
    self.assertEqual(output.shape, (batch_size,))


if __name__ == '__main__':
  absltest.main()
