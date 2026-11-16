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
"""Unit tests for programmatic validation utilities in stage4_antileak.py."""

from unittest import mock

from absl.testing import absltest

import stage4_antileak


class TimestampParsingTest(absltest.TestCase):

  def test_parse_timestamp_range_valid(self):
    text = 'Between 0:12 and 0:15 in the clip'
    res = stage4_antileak._parse_timestamp_range(text)
    self.assertIsNotNone(res)
    assert res is not None
    self.assertEqual(res, (12.0, 15.0))

  def test_parse_timestamp_range_invalid(self):
    text = 'No timestamps here'
    res = stage4_antileak._parse_timestamp_range(text)
    self.assertIsNone(res)

  def test_parse_timestamp_range_fractional_seconds(self):
    text = 'Between 1:05.5 and 2:10.3'
    res = stage4_antileak._parse_timestamp_range(text)
    self.assertIsNotNone(res)
    assert res is not None
    self.assertAlmostEqual(res[0], 65.5)
    self.assertAlmostEqual(res[1], 130.3)


class TemporalOverlapTest(absltest.TestCase):

  def test_detects_overlap(self):
    gt = 'Between 0:10 and 0:15'
    distractor = 'Between 0:14 and 0:20'
    self.assertTrue(stage4_antileak._is_temporal_overlap(gt, distractor))

  def test_non_overlapping(self):
    gt = 'Between 0:10 and 0:15'
    distractor = 'Between 0:20 and 0:25'
    self.assertFalse(stage4_antileak._is_temporal_overlap(gt, distractor))

  def test_exceeds_clip_duration(self):
    gt = 'Between 0:02 and 0:05'
    distractor = 'Between 0:25 and 0:35'
    self.assertTrue(
        stage4_antileak._is_temporal_overlap(gt, distractor, max_clip_sec=30.0)
    )

  def test_no_timestamp_in_distractor(self):
    gt = 'Between 0:10 and 0:15'
    distractor = 'Redness on the left hand'
    self.assertFalse(stage4_antileak._is_temporal_overlap(gt, distractor))


def _make_audit_result(audits):
  """Creates a mock ItemCollisionAuditBatch from (text, is_collision) pairs."""
  mock_batch = mock.MagicMock()
  mock_audits = []
  for text, is_collision in audits:
    a = mock.MagicMock()
    a.distractor_text = text
    a.is_valid_answer_to_stem = is_collision
    mock_audits.append(a)
  mock_batch.audits = mock_audits
  return mock_batch


class RunCollisionAuditTest(absltest.TestCase):

  def test_returns_none_when_all_collide(self):
    lm = mock.MagicMock()
    # All 9 distractors are collisions every retry
    audit_res = _make_audit_result([(f'd{i}', True) for i in range(9)])
    lm.generate_thought_and_object.return_value = (audit_res, None)

    result = stage4_antileak._run_collision_audit(
        question='Q',
        gt_answer='A',
        distractors=[f'd{i}' for i in range(9)],
        clip_path='/tmp/clip.mp4',
        clip_dur_sec=10.0,
        idx=0,
        lm=lm,
    )
    self.assertIsNone(result)

  @mock.patch.object(stage4_antileak, 'content_utils')
  def test_returns_clean_distractors(self, mock_content_utils):
    lm = mock.MagicMock()
    # All 9 distractors pass audit
    audit_res = _make_audit_result(
        [(f'distractor_{i}', False) for i in range(9)]
    )
    lm.generate_thought_and_object.return_value = (audit_res, None)

    result = stage4_antileak._run_collision_audit(
        question='Q',
        gt_answer='A',
        distractors=[f'distractor_{i}' for i in range(9)],
        clip_path='/tmp/clip.mp4',
        clip_dur_sec=10.0,
        idx=0,
        lm=lm,
    )
    self.assertIsNotNone(result)
    self.assertLen(result, 9)
    self.assertTrue(mock_content_utils.media_part.called)

  @mock.patch.object(stage4_antileak, 'content_utils')
  def test_retry_does_not_reaudit_rejected_distractors(self, unused_utils):
    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = [
        # First audit: one collision, eight clean distractors.
        (
            _make_audit_result(
                [('colliding answer', True)]
                + [(f'distractor_{i}', False) for i in range(8)]
            ),
            None,
        ),
        # One replacement distractor.
        (mock.MagicMock(distractors=['new distractor']), None),
        # Second audit.
        (_make_audit_result([('new distractor', False)]), None),
    ]

    result = stage4_antileak._run_collision_audit(
        question='Q',
        gt_answer='A',
        distractors=['colliding answer']
        + [f'distractor_{i}' for i in range(8)],
        clip_path='/tmp/clip.mp4',
        clip_dur_sec=10.0,
        idx=0,
        lm=lm,
    )

    self.assertLen(result, 9)
    self.assertNotIn('colliding answer', result)
    second_audit_prompt = lm.generate_thought_and_object.call_args_list[2][0][
        0
    ][1]
    self.assertIn('new distractor', second_audit_prompt)
    self.assertNotIn('colliding answer', second_audit_prompt)

  def test_handles_lm_exception(self):
    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = RuntimeError('LM failed')

    result = stage4_antileak._run_collision_audit(
        question='Q',
        gt_answer='A',
        distractors=[f'd{i}' for i in range(9)],
        clip_path='/tmp/clip.mp4',
        clip_dur_sec=10.0,
        idx=0,
        lm=lm,
    )
    self.assertIsNone(result)

  @mock.patch.object(stage4_antileak, 'content_utils')
  def test_custom_distractor_count_needs_no_replacements(
      self, unused_mock_content_utils
  ):
    lm = mock.MagicMock()
    audit_res = _make_audit_result(
        [(f'distractor_{i}', False) for i in range(3)]
    )
    lm.generate_thought_and_object.return_value = (audit_res, None)

    result = stage4_antileak._run_collision_audit(
        question='Q',
        gt_answer='A',
        distractors=[f'distractor_{i}' for i in range(3)],
        clip_path='/tmp/clip.mp4',
        clip_dur_sec=10.0,
        idx=0,
        lm=lm,
        num_distractors=3,
    )
    self.assertEqual(result, ['distractor_0', 'distractor_1', 'distractor_2'])
    # Only the audit call; no replacement distractors were requested.
    lm.generate_thought_and_object.assert_called_once()


class RunAntileakGateTest(absltest.TestCase):

  def test_rejects_leaked_item(self):
    lm = mock.MagicMock()
    mock_answer = mock.MagicMock()
    # Use a real string so strip().upper() works naturally
    mock_answer.selected_option = 'A'
    lm.generate_thought_and_object.return_value = (mock_answer, None)

    distractors = [f'distractor_{i}' for i in range(9)]
    item = {
        'question': 'Q',
        'correct_answer': 'A',
        'distractors': distractors,
    }

    # With no shuffle, gt_answer stays at index 0 -> key 'A'
    with mock.patch('random.shuffle', side_effect=lambda x: None):
      result = stage4_antileak._run_antileak_gate(
          question='Q',
          gt_answer='A',
          collision_free=distractors,
          idx=0,
          lm=lm,
          item=item,
      )
    self.assertIsNone(result)

  def test_passes_non_leaked_item(self):
    lm = mock.MagicMock()
    mock_answer = mock.MagicMock()
    # Model picks wrong answer (not leaked)
    mock_answer.selected_option = 'Z'
    lm.generate_thought_and_object.return_value = (mock_answer, None)

    distractors = [f'distractor_{i}' for i in range(9)]
    item = {
        'question': 'Q',
        'correct_answer': 'A',
        'distractors': distractors,
    }

    result = stage4_antileak._run_antileak_gate(
        question='Q',
        gt_answer='A',
        collision_free=distractors,
        idx=0,
        lm=lm,
        item=item,
    )
    self.assertIsNotNone(result)
    self.assertEqual(result['collision_audit_status'], 'PASS')
    self.assertEqual(result['gate_status'], 'PASS')
    self.assertIn('options', result)
    self.assertIn('correct_key', result)

  def test_rejects_item_when_lm_fails(self):
    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = RuntimeError('LM failed')
    distractors = [f'distractor_{i}' for i in range(9)]

    result = stage4_antileak._run_antileak_gate(
        question='Q',
        gt_answer='A',
        collision_free=distractors,
        idx=0,
        lm=lm,
        item={'question': 'Q', 'correct_answer': 'A'},
    )
    self.assertIsNone(result)

  def test_default_prompt_has_ten_options(self):
    lm = mock.MagicMock()
    mock_answer = mock.MagicMock()
    mock_answer.selected_option = 'Z'
    lm.generate_thought_and_object.return_value = (mock_answer, None)

    result = stage4_antileak._run_antileak_gate(
        question='Q',
        gt_answer='A',
        collision_free=[f'distractor_{i}' for i in range(12)],
        idx=0,
        lm=lm,
        item={'question': 'Q', 'correct_answer': 'A'},
    )

    prompt = lm.generate_thought_and_object.call_args[0][0]
    self.assertIn('Answer this 10-option clinical question', prompt)
    self.assertIn('Select the single letter (A-J)', prompt)
    self.assertEqual(
        [opt['key'] for opt in result['options']], list('ABCDEFGHIJ')
    )
    self.assertEqual(result['num_options'], 10)

  def test_respects_custom_distractor_count(self):
    lm = mock.MagicMock()
    mock_answer = mock.MagicMock()
    mock_answer.selected_option = 'Z'
    lm.generate_thought_and_object.return_value = (mock_answer, None)

    with mock.patch('random.shuffle', side_effect=lambda x: None):
      result = stage4_antileak._run_antileak_gate(
          question='Q',
          gt_answer='A',
          collision_free=[f'distractor_{i}' for i in range(5)],
          idx=0,
          lm=lm,
          item={'question': 'Q', 'correct_answer': 'A'},
          num_distractors=3,
      )

    prompt = lm.generate_thought_and_object.call_args[0][0]
    self.assertIn('Answer this 4-option clinical question', prompt)
    self.assertIn('Select the single letter (A-D)', prompt)
    self.assertIn('D. distractor_2', prompt)
    self.assertNotIn('distractor_3', prompt)
    self.assertEqual([opt['key'] for opt in result['options']], list('ABCD'))
    self.assertEqual(result['correct_key'], 'A')
    self.assertEqual(result['num_options'], 4)


class ProcessItemTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.clips_dir = self.create_tempdir()
    self.item = {
        'question': 'Q',
        'correct_answer': 'A',
        'distractors': [f'distractor_{i}' for i in range(3)],
        'clip_file': 'clip_01.mp4',
        'clip_duration_sec': 10.0,
        'num_distractors': 3,
    }

  def test_missing_clip_returns_none_without_model_calls(self):
    lm = mock.MagicMock()

    result = stage4_antileak._process_item(
        0, self.item, lm, self.clips_dir.full_path
    )

    self.assertIsNone(result)
    lm.generate_thought_and_object.assert_not_called()

  def test_uses_muted_clip_and_item_distractor_count(self):
    self.clips_dir.create_file('clip_01_noaudio.mp4', content='muted-clip')
    audit = _make_audit_result([(f'distractor_{i}', False) for i in range(3)])
    answer = mock.MagicMock()
    answer.selected_option = 'Z'
    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = [(audit, None), (answer, None)]

    result = stage4_antileak._process_item(
        0, self.item, lm, self.clips_dir.full_path
    )

    self.assertEqual(result['num_options'], 4)
    audit_contents = lm.generate_thought_and_object.call_args_list[0][0][0]
    self.assertEqual(audit_contents[0]['inline_data']['data'], b'muted-clip')


class GenerateReplacementDistractorsTest(absltest.TestCase):

  def test_adds_to_candidate_pool(self):
    lm = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.distractors = ['new_d1', 'new_d2']
    lm.generate_thought_and_object.return_value = (mock_result, None)

    candidate_pool = ['existing_d']
    stage4_antileak._generate_replacement_distractors(
        question='Q',
        gt_answer='A',
        collision_free=[],
        candidate_pool=candidate_pool,
        needed=2,
        lm=lm,
    )
    self.assertIn('new_d1', candidate_pool)
    self.assertIn('new_d2', candidate_pool)

  def test_skips_duplicates(self):
    lm = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.distractors = ['existing_d', 'new_d']
    lm.generate_thought_and_object.return_value = (mock_result, None)

    candidate_pool = ['existing_d']
    stage4_antileak._generate_replacement_distractors(
        question='Q',
        gt_answer='A',
        collision_free=[],
        candidate_pool=candidate_pool,
        needed=2,
        lm=lm,
    )
    self.assertEqual(candidate_pool.count('existing_d'), 1)
    self.assertIn('new_d', candidate_pool)

  def test_handles_lm_exception(self):
    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = RuntimeError('fail')

    candidate_pool = ['d1']
    stage4_antileak._generate_replacement_distractors(
        question='Q',
        gt_answer='A',
        collision_free=[],
        candidate_pool=candidate_pool,
        needed=2,
        lm=lm,
    )
    # Pool should be unchanged
    self.assertEqual(candidate_pool, ['d1'])


if __name__ == '__main__':
  absltest.main()
