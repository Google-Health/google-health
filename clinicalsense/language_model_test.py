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
"""Unit tests for the Gemini API wrapper."""

from unittest import mock

from absl.testing import absltest
import pydantic

import constraints
import language_model


@pydantic.dataclasses.dataclass
class _Answer(constraints.BaseModel):
  """Test schema."""

  selected_option: str


class GenerateThoughtAndObjectTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    client_cls = self.enter_context(
        mock.patch.object(language_model.genai, 'Client')
    )
    self.client = client_cls.return_value
    self.response_text = '```json\n{"selected_option": "C"}\n```'
    self.client.models.generate_content.return_value = mock.Mock(
        text=self.response_text
    )
    self.lm = language_model.LanguageModel(
        model_name='test-model', api_key='test-key'
    )

  def test_text_prompt_is_sent_unchanged(self):
    prompt = 'Pick an option.' + constraints.get_constraint_suffix(cls=_Answer)

    answer, raw_response = self.lm.generate_thought_and_object(
        prompt, cls=_Answer
    )

    self.assertEqual(answer.selected_option, 'C')
    self.assertEqual(raw_response, self.response_text)
    self.client.models.generate_content.assert_called_once_with(
        model='test-model', contents=prompt
    )

  def test_multimodal_prompt_is_sent_unchanged(self):
    contents = [
        {'inline_data': {'mime_type': 'video/mp4', 'data': b'video'}},
        'Pick an option.' + constraints.get_constraint_suffix(cls=_Answer),
    ]

    answer, _ = self.lm.generate_thought_and_object(contents, cls=_Answer)

    self.assertEqual(answer.selected_option, 'C')
    self.client.models.generate_content.assert_called_once_with(
        model='test-model', contents=contents
    )


if __name__ == '__main__':
  absltest.main()
