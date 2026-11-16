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
"""Unit tests for Stage 2 question generation."""

from unittest import mock

from absl.testing import absltest

import stage2_question_gen


class Stage2QuestionGenTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.facet_instructions = stage2_question_gen._build_facet_instructions()
    self.sample_clip_item = {
        'clip': 'clip_01.mp4',
        'case_id': 'case_101',
        'condition': 'Rotator Cuff Tear',
        'symptom': 'Shoulder pain',
        'clip_duration_sec': 15.0,
        'clip_duration_str': '00:15',
        'evidence': {
            'primary_sign': 'Patient rubs right shoulder',
            'action': 'Guarding right shoulder',
            'spatial': 'Right anterior shoulder',
            'temporal': 'Between 0:05 and 0:10',
            'appearance': 'Grimacing with eyes closed',
            'absence': 'No respiratory distress',
        },
    }

  def test_prompt_template_loads_and_renders(self):
    template = stage2_question_gen._load_prompt_template()
    evidence = self.sample_clip_item['evidence']
    rendered = template.render(
        primary_sign=evidence['primary_sign'],
        action=evidence['action'],
        spatial=evidence['spatial'],
        temporal=evidence['temporal'],
        appearance=evidence['appearance'],
        absence=evidence['absence'],
        num_facets=5,
        facet_instructions=self.facet_instructions,
    )
    self.assertIn('Primary sign: Patient rubs right shoulder', rendered)
    self.assertIn('ACTION (Intentionality): Guarding right shoulder', rendered)
    self.assertIn('Generate exactly 5 questions', rendered)

  def test_build_facet_instructions_contains_all_facets(self):
    required_facets = [
        'ACTION',
        'SPATIAL',
        'TEMPORAL',
        'APPEARANCE',
        'ABSENCE',
    ]
    for facet in required_facets:
      self.assertIn(f'[{facet}]', self.facet_instructions)

  def test_build_facet_instructions_numbered(self):
    for i in range(1, 6):
      self.assertIn(f'{i}. [', self.facet_instructions)

  def test_process_clip_returns_none_on_missing_evidence(self):
    lm_mock = mock.MagicMock()
    item = {'clip': 'clip_01.mp4'}
    result = stage2_question_gen._process_clip(
        item, lm_mock, self.facet_instructions
    )
    self.assertIsNone(result)

  def test_process_clip_returns_none_on_missing_clip(self):
    lm_mock = mock.MagicMock()
    item = {'evidence': {'primary_sign': 'Pain'}}
    result = stage2_question_gen._process_clip(
        item, lm_mock, self.facet_instructions
    )
    self.assertIsNone(result)

  def test_process_clip_returns_none_on_lm_error(self):
    lm_mock = mock.MagicMock()
    lm_mock.generate_thought_and_object.side_effect = RuntimeError('LM failure')
    result = stage2_question_gen._process_clip(
        self.sample_clip_item, lm_mock, self.facet_instructions
    )
    self.assertIsNone(result)

  def test_process_clip_success(self):
    lm_mock = mock.MagicMock()
    mock_questions = [
        stage2_question_gen.GeneratedQuestion(
            eim_facet='ACTION',
            target_anchor='right arm',
            question='What guarding behavior is shown?',
            correct_answer='Rubs right shoulder',
            evidence_basis='Patient touches shoulder',
        ),
        stage2_question_gen.GeneratedQuestion(
            eim_facet='SPATIAL',
            target_anchor='shoulder',
            question='Which shoulder is affected?',
            correct_answer='Right shoulder',
            evidence_basis='Touches right side',
        ),
    ]
    batch = stage2_question_gen.QuestionBatch(questions=mock_questions)
    lm_mock.generate_thought_and_object.return_value = (batch, None)

    result = stage2_question_gen._process_clip(
        self.sample_clip_item, lm_mock, self.facet_instructions
    )

    self.assertIsNotNone(result)
    self.assertEqual(result['clip'], 'clip_01.mp4')
    self.assertEqual(result['case_id'], 'case_101')
    self.assertEqual(result['condition'], 'Rotator Cuff Tear')
    self.assertEqual(result['symptom'], 'Shoulder pain')
    self.assertEqual(result['clip_duration_sec'], 15.0)
    self.assertEqual(result['clip_duration_str'], '00:15')
    self.assertLen(result['questions'], 2)
    self.assertEqual(result['questions'][0]['eim_facet'], 'ACTION')
    self.assertEqual(
        result['questions'][0]['question'], 'What guarding behavior is shown?'
    )
    self.assertEqual(
        result['questions'][0]['correct_answer'], 'Rubs right shoulder'
    )


if __name__ == '__main__':
  absltest.main()
