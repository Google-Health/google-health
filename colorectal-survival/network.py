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
"""Network."""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import TimeDistributed


def build_network(input_shape,
                  base_depth=16,
                  depth_growth=1.25,
                  stride_2_layers=4,
                  stride_1_layers=1,
                  kernel_size=3):
  """Returns deep learning network for survival prediction.

  The network takes a set of image patches as input. A feature vector
  is extracted from each image patch using a CNN module. Feature vectors are
  averaged before being fed into a dense layer with a single output.

  Args:
    input_shape: 4D input shape [sequence_length, height, width, depth].
    base_depth: the number of filter in the base Conv2D layer.
    depth_growth: the rate at which the number of channels in the feature map
      grows after each layer with stride 2.
    stride_2_layers: the number of SeparableConv2D layers with stride 2 layer.
    stride_1_layers: the number of SeparableConv2D layers with stride 1 layers
      between each SeparableConv2D with stride 2 layer.
    kernel_size:  integer specifying the height and width of the 2D convolution
      window
  """

  if len(input_shape) != 4:
    raise ValueError('Expecting 4D input')

  model = tf.keras.Sequential()

  # Create CNN for image patch feature extraction
  cnn = build_cnn(input_shape[1:], base_depth, depth_growth, stride_2_layers,
                  stride_1_layers, kernel_size)

  # Run CNN on each image patch
  model.add(TimeDistributed(cnn, input_shape=input_shape))

  # Copmute average of image patch features per example
  model.add(GlobalAveragePooling1D())

  # Compute risk scores
  model.add(Dense(1))

  return model


def build_cnn(input_shape, base_depth, depth_growth, stride_2_layers,
              stride_1_layers, kernel_size):
  """Returns CNN for extrating image patch features.

  Args:
    input_shape: 3D shape of input image patches.
    base_depth: the number of filter in the base Conv2D layer.
    depth_growth: the rate at which the number of channels in the feature map
      grows after each layer with stride 2.
    stride_2_layers: the number of SeparableConv2D layers with stride 2 layer.
    stride_1_layers: the number of SeparableConv2D layers with stride 1 layers
      between each SeparableConv2D with stride 2 layer.
    kernel_size:  integer specifying the height and width of the 2D convolution
      window
  """
  model = tf.keras.Sequential()

  # Configure base layer
  base_layer = Conv2D(
      base_depth,
      kernel_size,
      strides=kernel_size,
      activation='relu',
      padding='same',
      input_shape=input_shape)
  model.add(base_layer)
  model.add(BatchNormalization())

  # Depthwise separable convolution sequence
  for i in range(stride_2_layers):
    depth = int(base_depth * depth_growth**i)
    model.add(
        SeparableConv2D(
            depth, kernel_size, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    for _ in range(stride_1_layers):
      model.add(
          SeparableConv2D(
              depth, kernel_size, strides=1, activation='relu', padding='same'))
      model.add(BatchNormalization())

  # Spatial Pooling
  model.add(GlobalAveragePooling2D())

  return model
