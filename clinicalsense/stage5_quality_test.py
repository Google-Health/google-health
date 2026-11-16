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
"""Unit tests for final benchmark item formatting in stage5_quality.py."""

from absl.testing import absltest
import stage5_quality


class FormatBenchmarkItemTest(absltest.TestCase):

  def test_format_benchmark_item_valid(self):
    item = {
        'clip_file': 'clip_01.mp4',
        'case_id': '101',
        'condition': 'Angina',
        'symptom': 'Chest pain',
        'clip_duration_sec': 15.0,
        'eim_facet': 'ACTION',
        'question_type': 'action',
        'target_anchor': 'Hand placement',
        'question': 'Which action occurs?',
        'correct_answer': 'Presses chest',
        'evidence_basis': 'Visible pressing motion',
        'options': [
            {'key': 'A', 'text': 'Presses chest', 'is_correct': True},
            {'key': 'B', 'text': 'Rubs leg', 'is_correct': False},
        ],
        'correct_key': 'A',
        'domain_category': 'THORACIC',
    }
    res = stage5_quality._format_benchmark_item(item, 1)
    self.assertIsNotNone(res)
    assert res is not None
    self.assertEqual(res['item_id'], 'CS-ACTION-001')
    self.assertEqual(res['correct_key'], 'A')
    self.assertEqual(res['num_options'], 2)

  def test_format_benchmark_item_missing_correct_answer(self):
    item = {'clip_file': 'clip_01.mp4'}
    res = stage5_quality._format_benchmark_item(item, 1)
    self.assertIsNone(res)

  def test_format_benchmark_item_fallback_options(self):
    item = {
        'clip_file': 'clip_01.mp4',
        'case_id': '101',
        'condition': 'Angina',
        'question': 'Which action occurs?',
        'correct_answer': 'Presses chest',
        'distractors': ['Rubs leg', 'Taps foot'],
    }
    res = stage5_quality._format_benchmark_item(item, 5)
    self.assertIsNotNone(res)
    assert res is not None
    self.assertEqual(res['item_id'], 'CS-ACTION-005')
    self.assertLen(res['options'], 3)
    self.assertEqual(res['correct_key'], 'A')


class FilterByClipDurationTest(absltest.TestCase):

  def _make_items(self, durations):
    return [
        {'clip_duration_sec': d, 'name': f'item_{i}'}
        for i, d in enumerate(durations)
    ]

  def test_filters_short_clips(self):
    items = self._make_items([1.0, 5.0, 3.0, 2.5])
    result = stage5_quality._filter_by_clip_duration(items, 3.0)
    self.assertLen(result, 2)
    self.assertEqual(result[0]['clip_duration_sec'], 5.0)
    self.assertEqual(result[1]['clip_duration_sec'], 3.0)

  def test_keeps_all_when_all_above_min(self):
    items = self._make_items([10.0, 20.0, 15.0])
    result = stage5_quality._filter_by_clip_duration(items, 3.0)
    self.assertLen(result, 3)

  def test_removes_all_when_all_below_min(self):
    items = self._make_items([1.0, 2.0, 0.5])
    result = stage5_quality._filter_by_clip_duration(items, 5.0)
    self.assertEmpty(result)

  def test_no_filter_when_min_dur_zero(self):
    items = self._make_items([0.1, 0.5, 1.0])
    result = stage5_quality._filter_by_clip_duration(items, 0.0)
    self.assertLen(result, 3)

  def test_no_filter_when_min_dur_negative(self):
    items = self._make_items([0.1, 0.5])
    result = stage5_quality._filter_by_clip_duration(items, -1.0)
    self.assertLen(result, 2)

  def test_empty_input(self):
    result = stage5_quality._filter_by_clip_duration([], 3.0)
    self.assertEmpty(result)

  def test_missing_clip_duration_defaults_to_zero(self):
    items = [{'name': 'no_duration'}, {'clip_duration_sec': 5.0, 'name': 'ok'}]
    result = stage5_quality._filter_by_clip_duration(items, 3.0)
    self.assertLen(result, 1)
    self.assertEqual(result[0]['name'], 'ok')

  def test_exact_boundary(self):
    items = self._make_items([3.0])
    result = stage5_quality._filter_by_clip_duration(items, 3.0)
    self.assertLen(result, 1)


class BuildSummaryTest(absltest.TestCase):

  def test_builds_correct_summary(self):
    items = [
        {'eim_facet': 'ACTION', 'condition': 'Angina'},
        {'eim_facet': 'ACTION', 'condition': 'Fracture'},
        {'eim_facet': 'SPATIAL', 'condition': 'Angina'},
    ]
    summary = stage5_quality._build_summary(items, 3.0)
    self.assertEqual(summary['benchmark_name'], 'ClinicalSense')
    self.assertEqual(summary['total_items'], 3)
    self.assertEqual(summary['min_clip_duration_sec'], 3.0)
    self.assertEqual(summary['by_eim_facet']['ACTION'], 2)
    self.assertEqual(summary['by_eim_facet']['SPATIAL'], 1)
    self.assertEqual(summary['by_condition']['Angina'], 2)
    self.assertLen(summary['items'], 3)

  def test_empty_items(self):
    summary = stage5_quality._build_summary([], 0.0)
    self.assertEqual(summary['total_items'], 0)
    self.assertEmpty(summary['items'])


if __name__ == '__main__':
  absltest.main()
