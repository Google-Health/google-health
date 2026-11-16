# Copyright 2026 Google LLC. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
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
"""Shared Jinja2 template utilities for ClinicalSense pipeline stages.

Provides template resolution and rendering used across Stage 1, Stage 1B
(verify), and Stage 1B-patch scripts.
"""

import os
from typing import Any

import jinja2


def render_template(template_name: str, **kwargs: Any) -> str:
  """Renders a Jinja2 template from the ClinicalSense templates directory.

  Args:
    template_name: Base name of the template (without .jinja2 extension).
    **kwargs: Variables to pass to the Jinja2 template.

  Returns:
    Rendered template string.
  """
  template_path = os.path.join(
      os.path.dirname(__file__), 'templates', f'{template_name}.jinja2'
  )
  with open(template_path, 'r') as f:
    template_str = f.read()

  env = jinja2.Environment(
      undefined=jinja2.StrictUndefined,
      trim_blocks=True,
      lstrip_blocks=True,
      autoescape=False,
      keep_trailing_newline=True,
  )
  return env.from_string(template_str).render(**kwargs)
