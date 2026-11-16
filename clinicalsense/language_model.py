# Copyright 2026 Google LLC. All rights reserved.
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

"""To use this module, install google-genai and set GOOGLE_API_KEY."""

import os

from google import genai
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

import constraints

# Model used to produce the results reported in the paper. Any model ID accepted
# by the Gemini API can be passed to LanguageModel or via the --model_name flag
# on each stage; see https://ai.google.dev/gemini-api/docs/models for the
# current list.
DEFAULT_MODEL = 'gemini-3.1-pro-preview'


class LanguageModel:
  """Wrapper around the Google Generative AI SDK for LLM calls."""

  def __init__(self, model_name=DEFAULT_MODEL, api_key=None):
    if api_key is None:
      api_key = os.environ.get('GOOGLE_API_KEY')
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name

  def generate(self, prompt, options=None):  # pylint: disable=unused-argument
    """Generates text from a prompt."""
    response = self.client.models.generate_content(
        model=self.model_name, contents=prompt
    )
    return response.text

  @retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(multiplier=1, min=4, max=10),
  )
  def generate_thought_and_object(  # pylint: disable=unused-argument
      self, prompt, cls, options=None, serializer='json'
  ):
    """Generates a structured object of type `cls` from a prompt.

    The request is not constrained to `cls`, so the prompt must already ask for
    the answer in that format: callers append
    `constraints.get_constraint_suffix(cls=cls)` to the text of the prompt.

    Args:
      prompt: The prompt text, or a list of parts (e.g. a video part from
        `content_utils` followed by the prompt text).
      cls: The pydantic dataclass to parse the response into.
      options: Unused; kept for API compatibility.
      serializer: Serialization format of the response. Only 'json' is
        supported.

    Returns:
      A tuple of the parsed `cls` object and the raw response text.
    """
    response = self.client.models.generate_content(
        model=self.model_name, contents=prompt
    )
    parsed_object = constraints.parse_object(
        response.text, cls, serializer=serializer
    )
    return parsed_object, response.text

  def generate_thought_and_response(self, prompt, options=None):
    """Generates a text response and returns it as a (response, thought) tuple."""
    response_text = self.generate(prompt, options=options)
    return response_text, ''
