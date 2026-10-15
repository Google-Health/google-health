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

"""Configuration classes for model used in the anemia screening AI model."""

import transformers


class ResNetMultimodalConfig(transformers.PretrainedConfig):
  """Configuration class for ResNetMultimodalModel.

  This class is used to store the configuration of a ResNet-based model
  that can handle both image and metadata inputs.
  """
  model_type = "resnet_multimodal"

  def __init__(
      self,
      num_channels: int = 3,
      embedding_size: int = 128,
      hidden_sizes: list[int] | None = None,
      depths: list[int] | None = None,
      layer_type: str = "basic",
      metadata_size: int = 3,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.num_channels = num_channels
    self.embedding_size = embedding_size
    self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [128, 256]
    self.depths = depths if depths is not None else [1, 1]
    self.layer_type = layer_type
    self.metadata_size = metadata_size
