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
"""Unit tests for Stage 3 distractor generation."""

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver

import stage3_distractors


class Stage3DistractorsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sample_clip_item = {
        'clip': 'clip_01.mp4',
        'case_id': 'case_101',
        'condition': 'Rotator Cuff Tear',
        'symptom': 'Shoulder pain',
        'clip_duration_sec': 30.0,
        'clip_duration_str': '00:30',
    }
    self.sample_question = {
        'question': 'Which shoulder is the patient guarding?',
        'correct_answer': 'Right shoulder',
        'eim_facet': 'SPATIAL',
        'question_type': 'spatial',
        'target_anchor': 'shoulder',
        'evidence_basis': 'Patient places left hand over right shoulder',
    }

  def test_prompt_template_loads_and_renders(self):
    template = stage3_distractors._load_prompt_template()
    rendered = template.render(
        question=self.sample_question['question'],
        correct_answer=self.sample_question['correct_answer'],
        eim_facet=self.sample_question['eim_facet'],
        target_anchor=self.sample_question['target_anchor'],
        evidence_basis=self.sample_question['evidence_basis'],
        clip_duration='00:30',
        num_distractors=9,
        num_options=10,
    )
    self.assertIn(
        'QUESTION STEM: Which shoulder is the patient guarding?', rendered
    )
    self.assertIn('GROUND TRUTH ANSWER: Right shoulder', rendered)
    self.assertIn('EIM FACET: SPATIAL', rendered)
    self.assertIn('CLIP DURATION: 00:30', rendered)
    self.assertIn('10-option multiple choice question', rendered)
    self.assertIn('EXACTLY 9 hard-negative distractors', rendered)

  def test_prompt_template_respects_custom_distractor_count(self):
    template = stage3_distractors._load_prompt_template()
    rendered = template.render(
        question=self.sample_question['question'],
        correct_answer=self.sample_question['correct_answer'],
        eim_facet=self.sample_question['eim_facet'],
        target_anchor=self.sample_question['target_anchor'],
        evidence_basis=self.sample_question['evidence_basis'],
        clip_duration='00:30',
        num_distractors=3,
        num_options=4,
    )
    self.assertIn('4-option multiple choice question', rendered)
    self.assertIn('EXACTLY 3 hard-negative distractors', rendered)
    self.assertIn('Output all 3 distractors', rendered)

  def test_process_question_returns_none_on_missing_question(self):
    lm_mock = mock.MagicMock()
    q = {'correct_answer': 'Right shoulder'}
    result = stage3_distractors._process_question(
        (self.sample_clip_item, q), lm_mock, num_distractors=9
    )
    self.assertIsNone(result)

  def test_process_question_returns_none_on_missing_answer(self):
    lm_mock = mock.MagicMock()
    q = {'question': 'Which shoulder is guarded?'}
    result = stage3_distractors._process_question(
        (self.sample_clip_item, q), lm_mock, num_distractors=9
    )
    self.assertIsNone(result)

  def test_process_question_returns_none_on_empty_question(self):
    lm_mock = mock.MagicMock()
    q = {'question': '', 'correct_answer': 'Right shoulder'}
    result = stage3_distractors._process_question(
        (self.sample_clip_item, q), lm_mock, num_distractors=9
    )
    self.assertIsNone(result)

  def test_process_question_returns_none_on_lm_error(self):
    lm_mock = mock.MagicMock()
    lm_mock.generate_thought_and_object.side_effect = RuntimeError('LM failure')
    result = stage3_distractors._process_question(
        (self.sample_clip_item, self.sample_question),
        lm_mock,
        num_distractors=9,
    )
    self.assertIsNone(result)

  def test_process_question_success(self):
    lm_mock = mock.MagicMock()
    mock_distractors = [
        stage3_distractors.CoHyponymDistractorCandidate(
            distractor_text=f'Distractor option {i}'
        )
        for i in range(1, 10)
    ]
    gen_result = stage3_distractors.CoHyponymGenerationResult(
        domain_category='SHOULDER_ARM',
        distractors=mock_distractors,
    )
    lm_mock.generate_thought_and_object.return_value = (gen_result, None)

    result = stage3_distractors._process_question(
        (self.sample_clip_item, self.sample_question),
        lm_mock,
        num_distractors=9,
    )

    self.assertIsNotNone(result)
    self.assertEqual(result['clip_file'], 'clip_01.mp4')
    self.assertEqual(result['case_id'], 'case_101')
    self.assertEqual(result['condition'], 'Rotator Cuff Tear')
    self.assertEqual(result['domain_category'], 'SHOULDER_ARM')
    self.assertEqual(result['num_options'], 10)
    self.assertEqual(result['num_distractors'], 9)
    self.assertLen(result['distractors'], 9)
    self.assertEqual(result['distractors'][0], 'Distractor option 1')

  def test_process_question_absence_facet_uses_video(self):
    clips_dir = self.create_tempdir()
    clips_dir.create_file('clip_01.mp4', content='video-bytes')
    lm_mock = mock.MagicMock()
    mock_distractors = [
        stage3_distractors.CoHyponymDistractorCandidate(
            distractor_text=f'Distractor option {i}'
        )
        for i in range(1, 10)
    ]
    gen_result = stage3_distractors.CoHyponymGenerationResult(
        domain_category='RESPIRATORY',
        distractors=mock_distractors,
    )
    lm_mock.generate_thought_and_object.return_value = (gen_result, None)

    absence_q = {
        'question': 'Which respiratory finding is NOT observed?',
        'correct_answer': 'Accessory muscle use',
        'eim_facet': 'ABSENCE',
        'target_anchor': 'chest',
        'evidence_basis': 'Normal chest excursion',
    }

    with flagsaver.flagsaver(clips_dir=clips_dir.full_path), mock.patch.object(
        stage3_distractors.content_utils,
        'video_chunk',
        return_value='video_chunk',
    ):
      result = stage3_distractors._process_question(
          (self.sample_clip_item, absence_q), lm_mock, num_distractors=9
      )

    self.assertIsNotNone(result)
    lm_mock.generate_thought_and_object.assert_called_once()
    content, _ = lm_mock.generate_thought_and_object.call_args
    # ABSENCE questions are video-grounded: [video part, prompt].
    self.assertEqual(content[0][0], 'video_chunk')
    self.assertIn('Which respiratory finding is NOT observed?', content[0][1])

  def test_build_work_items_flattens_questions(self):
    clips = [
        {
            'clip': 'c1.mp4',
            'questions': [{'question': 'q1'}, {'question': 'q2'}],
        },
        {
            'clip': 'c2.mp4',
            'questions': [{'question': 'q3'}],
        },
    ]
    work_items = stage3_distractors._build_work_items(clips)
    self.assertLen(work_items, 3)
    self.assertEqual(work_items[0][0]['clip'], 'c1.mp4')
    self.assertEqual(work_items[0][1]['question'], 'q1')
    self.assertEqual(work_items[1][1]['question'], 'q2')
    self.assertEqual(work_items[2][0]['clip'], 'c2.mp4')
    self.assertEqual(work_items[2][1]['question'], 'q3')

  def test_build_work_items_empty(self):
    work_items = stage3_distractors._build_work_items([])
    self.assertEmpty(work_items)


if __name__ == '__main__':
  absltest.main()
