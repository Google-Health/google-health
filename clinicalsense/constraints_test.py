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
"""Unit tests for structured-output parsing."""

from absl.testing import absltest
import pydantic

import constraints


@pydantic.dataclasses.dataclass
class _Answer(constraints.BaseModel):
  """Test schema."""

  selected_option: str


_ANSWER_JSON = '{"selected_option": "B"}'


class ParseObjectTest(absltest.TestCase):

  def _parse(self, response: str) -> str:
    return constraints.parse_object(response, _Answer).selected_option

  def test_parses_json_fenced_block(self):
    self.assertEqual(self._parse(f'```json\n{_ANSWER_JSON}\n```'), 'B')

  def test_parses_uppercase_json_fenced_block(self):
    self.assertEqual(self._parse(f'```JSON\n{_ANSWER_JSON}\n```'), 'B')

  def test_parses_bare_fenced_block(self):
    self.assertEqual(self._parse(f'```\n{_ANSWER_JSON}\n```'), 'B')

  def test_parses_unfenced_json(self):
    self.assertEqual(self._parse(_ANSWER_JSON), 'B')

  def test_prefers_json_block_over_earlier_bare_block(self):
    response = f'```\nnot json\n```\nAnswer:\n```json\n{_ANSWER_JSON}\n```'
    self.assertEqual(self._parse(response), 'B')

  def test_invalid_json_raises(self):
    with self.assertRaises(constraints.ResponseParsingError):
      self._parse('```json\nnot json\n```')


if __name__ == '__main__':
  absltest.main()
