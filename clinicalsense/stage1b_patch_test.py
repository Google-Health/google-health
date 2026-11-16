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
"""Unit tests for Stage 1B-patch corrections and excisions."""

import re
from unittest import mock

from absl.testing import absltest

import stage1b_patch


def _flagged(facet, claim, correction='', confidence='HIGH'):
  return {
      'facet': facet,
      'claim': claim,
      'claim_type': 'ACTION',
      'is_supported': False,
      'confidence': confidence,
      'evidence': 'Not visible in the clip.',
      'suggested_correction': correction,
  }


def _item(clip, evidence, flagged):
  return {
      'clip': clip,
      'condition': 'Knee osteoarthritis',
      'evidence': evidence,
      'verification': {'flagged_claims': flagged},
  }


def _patched(text, changes=None):
  return (
      stage1b_patch.PatchedEvidence(
          corrected_text=text, changes_made=changes or ['edited']
      ),
      '',
  )


class PatchSingleItemTest(absltest.TestCase):

  def test_unflagged_item_is_returned_unchanged(self):
    lm = mock.MagicMock()
    item = _item('c.mp4', {'action': 'Patient sits.'}, [])
    patched, num_patches, was_patched, diff = stage1b_patch._patch_single_item(
        item, lm
    )
    self.assertEqual(patched['evidence'], {'action': 'Patient sits.'})
    self.assertNotIn('verification', patched)
    self.assertEqual(num_patches, 0)
    self.assertFalse(was_patched)
    self.assertEmpty(diff)
    lm.generate_thought_and_object.assert_not_called()

  def test_excises_claim_without_correction(self):
    lm = mock.MagicMock()
    lm.generate_thought_and_object.return_value = _patched(
        'Patient sits upright.', ['Removed the knee rubbing claim.']
    )
    item = _item(
        'c.mp4',
        {'action': 'Patient sits upright. Patient rubs the left knee.'},
        [_flagged('action', 'Patient rubs the left knee.')],
    )

    patched, num_patches, was_patched, diff = stage1b_patch._patch_single_item(
        item, lm
    )

    self.assertTrue(was_patched)
    self.assertEqual(num_patches, 1)
    self.assertEqual(patched['evidence']['action'], 'Patient sits upright.')
    prompt = lm.generate_thought_and_object.call_args[0][0]
    self.assertIn(
        '- FABRICATED (remove entirely): "Patient rubs the left knee."', prompt
    )
    self.assertIn('\n  FACET: action [EXCISION]', diff)
    self.assertIn('  AFTER:  Patient sits upright.', diff)
    self.assertIn('    ✂ Removed the knee rubbing claim.', diff)

  def test_excision_falls_back_to_substring_removal(self):
    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = RuntimeError('LM failure')
    item = _item(
        'c.mp4',
        {'action': 'Patient sits upright. Patient rubs the left knee.'},
        [_flagged('action', 'Patient rubs the left knee.')],
    )

    patched, num_patches, _, diff = stage1b_patch._patch_single_item(item, lm)

    self.assertEqual(num_patches, 1)
    self.assertEqual(patched['evidence']['action'], 'Patient sits upright.')
    self.assertIn('    ✂ Removed: "Patient rubs the left knee."', diff)

  def test_low_confidence_correction_is_not_applied(self):
    lm = mock.MagicMock()
    item = _item(
        'c.mp4',
        {'spatial': 'Left knee.'},
        [_flagged('spatial', 'Left knee.', 'Right knee.', confidence='LOW')],
    )
    patched, num_patches, was_patched, _ = stage1b_patch._patch_single_item(
        item, lm
    )
    self.assertFalse(was_patched)
    self.assertEqual(num_patches, 0)
    self.assertEqual(patched['evidence'], {'spatial': 'Left knee.'})
    lm.generate_thought_and_object.assert_not_called()

  def test_applies_corrections_before_excisions(self):
    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = [
        _patched('Right knee.'),
        _patched('Patient sits.'),
    ]
    item = _item(
        'c.mp4',
        {'spatial': 'Left knee.', 'action': 'Patient sits. Patient waves.'},
        [
            _flagged('action', 'Patient waves.'),
            _flagged('spatial', 'Left knee.', 'Right knee.'),
        ],
    )

    patched, num_patches, _, diff = stage1b_patch._patch_single_item(item, lm)

    self.assertEqual(num_patches, 2)
    self.assertEqual(
        patched['evidence'],
        {'spatial': 'Right knee.', 'action': 'Patient sits.'},
    )
    first_prompt = lm.generate_thought_and_object.call_args_list[0][0][0]
    self.assertIn('CORRECT: "Right knee."', first_prompt)
    self.assertLess(
        diff.index('\n  FACET: spatial'),
        diff.index('\n  FACET: action [EXCISION]'),
    )


class PatchAllItemsTest(absltest.TestCase):

  def test_diff_blocks_stay_with_their_item_and_facet(self):

    def _fake_patch(prompt, cls):
      del cls  # Unused.
      facet = re.search(r'ORIGINAL EVIDENCE TEXT \((\w+)\)', prompt).group(1)
      return _patched(f'{facet} text after excision')

    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = _fake_patch
    items = [
        _item(
            f'clip_{i}.mp4',
            {'temporal': 'T claim.', 'appearance': 'A claim.'},
            [
                _flagged('temporal', 'T claim.'),
                _flagged('appearance', 'A claim.'),
            ],
        )
        for i in range(3)
    ]

    patched_items, diff, num_items_patched, num_corrections = (
        stage1b_patch._patch_all_items(items, lm)
    )

    self.assertLen(patched_items, 3)
    self.assertEqual(num_items_patched, 3)
    self.assertEqual(num_corrections, 6)
    headers = [i for i, line in enumerate(diff) if line.startswith('===')]
    self.assertEqual(
        [diff[i] for i in headers],
        [f'=== clip_{i}.mp4 (Knee osteoarthritis) ===' for i in range(3)],
    )
    for i, line in enumerate(diff):
      if line.endswith('[EXCISION]'):
        facet = line.split()[1]
        self.assertEqual(diff[i + 2], f'  AFTER:  {facet} text after excision')


if __name__ == '__main__':
  absltest.main()
