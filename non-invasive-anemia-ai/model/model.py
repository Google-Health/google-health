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

"""Defines the ResNetMultimodalRegressor model for anemia detection."""

import math

import torch
from torch import nn
import torch.nn.functional as F
import transformers

from model import config as resnet_multimodal_config


def same_pad_2d(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
  """Compute and apply JAX SAME padding to input before conv."""
  in_h, in_w = x.shape[2], x.shape[3]

  # Compute padding for H
  out_h = math.ceil(in_h / stride)
  total_h = max((out_h - 1) * stride + kernel_size - in_h, 0)
  pad_top = total_h // 2
  pad_bottom = total_h - pad_top

  # Compute padding for W
  out_w = math.ceil(in_w / stride)
  total_w = max((out_w - 1) * stride + kernel_size - in_w, 0)
  pad_left = total_w // 2
  pad_right = total_w - pad_left

  # F.pad order is (left, right, top, bottom)
  return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class SamePadMaxPool(nn.Module):
  """Max-pooling layer implementing JAX-style SAME padding with -inf fill."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    in_h, in_w = x.shape[2], x.shape[3]
    kernel_size, stride = 3, 2
    pad_h = max(math.ceil(in_h / stride - 1) * stride + kernel_size - in_h, 0)
    pad_w = max(math.ceil(in_w / stride - 1) * stride + kernel_size - in_w, 0)
    x = F.pad(
        x,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        value=-float('inf'),
    )
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=0)


class ResNetMultimodalRegressor(transformers.PreTrainedModel):
  """A multimodal ResNet-based regressor for anemia detection.

  This model combines visual features extracted by a ResNet backbone with
  metadata to predict a continuous value.
  """
  config_class = resnet_multimodal_config.ResNetMultimodalConfig

  def __init__(self, config):
    super().__init__(config)

    # Create an internal ResNetConfig for the backbone
    resnet_backbone_config = transformers.ResNetConfig(
        num_channels=config.num_channels,
        embedding_size=config.embedding_size,
        hidden_sizes=config.hidden_sizes,
        depths=config.depths,
        layer_type=config.layer_type,
    )
    self.backbone = transformers.ResNetBackbone(resnet_backbone_config)

    # Update ResNetBackbone architecture
    self.backbone.embedder.embedder.convolution.padding = (0, 0)
    self.backbone.embedder.embedder.activation = nn.Identity()
    self.backbone.embedder.pooler = SamePadMaxPool()  # pyrefly: ignore[bad-assignment]

    self.backbone.encoder.stages[0].layers[0].layer[
        0
    ].convolution.padding = 'same'
    self.backbone.encoder.stages[0].layers[0].layer[
        1
    ].convolution.padding = 'same'

    self.backbone.encoder.stages[1].layers[0].layer[0].convolution.padding = (
        0,
        0,
    )

    conv1 = self.backbone.encoder.stages[1].layers[0].layer[0].convolution
    conv1.forward = lambda x: F.conv2d(
        same_pad_2d(x, kernel_size=3, stride=2),
        conv1.weight,
        stride=2,
        padding=0,
    )

    self.pooler = nn.AdaptiveAvgPool2d((1, 1))
    self.mlp_2 = nn.Sequential(
        nn.Linear(256 + config.metadata_size, 64), nn.ReLU()
    )
    self.mlp_1 = nn.Sequential(
        nn.Linear(64, 1),
    )

  def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
    """Forward pass combining input image and tabular metadata.

    Args:
      x: Image tensor of shape [batch_size, 3, height, width].
      metadata: Metadata tensor of shape [batch_size, metadata_size].

    Returns:
      Predicted continuous hemoglobin regression values of shape [batch_size].
    """
    x = same_pad_2d(x, kernel_size=7, stride=2)
    x = self.backbone(x)
    x = x.feature_maps[-1]

    x = torch.flatten(self.pooler(x), 1)

    x = torch.cat((x, metadata), dim=1)
    x = self.mlp_2(x)
    x = self.mlp_1(x).squeeze(-1)

    return x
