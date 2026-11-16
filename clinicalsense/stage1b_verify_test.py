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
"""Unit tests for Stage 1B claim verification and re-verification."""

import os
from unittest import mock

from absl.testing import absltest

import stage1b_verify

_CLAIM = 'Patient points to the left knee.'
_CLIP_PART = {'inline_data': {'mime_type': 'video/mp4', 'data': b'clip'}}


def _verdict(
    is_supported: bool,
    evidence: str = '',
    correction: str = '',
    confidence: str = 'HIGH',
) -> stage1b_verify.ClaimVerdict:
  if not evidence:
    evidence = 'Clearly shown.' if is_supported else 'The right knee is shown.'
  return stage1b_verify.ClaimVerdict(
      is_supported=is_supported,
      confidence=confidence,
      evidence=evidence,
      suggested_correction=correction,
  )


class VerifyClaimWithRechecksTest(absltest.TestCase):

  def _verify(self, side_effect, claim_type='ACTION', **config_kwargs):
    """Runs _verify_claim_with_rechecks with _verify_claim mocked out."""
    config = stage1b_verify.RecheckConfig(**config_kwargs)
    with mock.patch.object(
        stage1b_verify, '_verify_claim', side_effect=side_effect
    ) as mock_verify:
      verdict, info = stage1b_verify._verify_claim_with_rechecks(
          claim_text=_CLAIM,
          claim_type=claim_type,
          clip_part=_CLIP_PART,
          lm=mock.MagicMock(),
          config=config,
      )
    return verdict, info, mock_verify.call_count

  def test_supported_claim_gets_single_pass(self):
    verdict, info, num_calls = self._verify([_verdict(True)])
    self.assertEqual(num_calls, 1)
    self.assertTrue(verdict.is_supported)
    self.assertEqual(
        info,
        {
            'initial_is_supported': True,
            'rechecked': False,
            'recheck_votes_supported': 0,
            'recheck_votes_total': 0,
        },
    )

  def test_failed_claim_overturned_by_recheck_majority(self):
    verdict, info, num_calls = self._verify([
        _verdict(False, correction='right knee'),
        _verdict(True),
        _verdict(False),
        _verdict(True),
    ])
    self.assertEqual(num_calls, 4)
    self.assertTrue(verdict.is_supported)
    self.assertEqual(verdict.suggested_correction, '')
    self.assertEqual(verdict.confidence, 'MEDIUM')
    self.assertStartsWith(verdict.evidence, '[Recheck majority 2/3]')
    self.assertFalse(info['initial_is_supported'])
    self.assertTrue(info['rechecked'])
    self.assertEqual(info['recheck_votes_supported'], 2)
    self.assertEqual(info['recheck_votes_total'], 3)

  def test_failed_claim_confirmed_by_recheck_majority(self):
    verdict, info, num_calls = self._verify([
        _verdict(False, correction='Patient points to the right knee.'),
        _verdict(False, correction='Patient points to the right knee.'),
        _verdict(True),
        _verdict(False, correction='Patient points to the right knee.'),
    ])
    self.assertEqual(num_calls, 4)
    self.assertFalse(verdict.is_supported)
    self.assertEqual(
        verdict.suggested_correction, 'Patient points to the right knee.'
    )
    self.assertEqual(info['recheck_votes_supported'], 1)
    self.assertEqual(info['recheck_votes_total'], 3)

  def test_supported_laterality_claim_is_still_rechecked(self):
    verdict, info, num_calls = self._verify(
        [_verdict(True), _verdict(False), _verdict(False), _verdict(True)],
        claim_type='LATERALITY',
    )
    self.assertEqual(num_calls, 4)
    self.assertFalse(verdict.is_supported)
    self.assertTrue(info['initial_is_supported'])
    self.assertTrue(info['rechecked'])
    self.assertEqual(info['recheck_votes_supported'], 1)

  def test_unanimous_rechecks_keep_best_confidence(self):
    verdict, _, _ = self._verify(
        [
            _verdict(True),
            _verdict(True, confidence='MEDIUM'),
            _verdict(True, confidence='HIGH'),
            _verdict(True, confidence='LOW'),
        ],
        claim_type='LATERALITY',
    )
    self.assertTrue(verdict.is_supported)
    self.assertEqual(verdict.confidence, 'HIGH')

  def test_recheck_tie_is_broken_by_initial_verdict(self):
    verdict, info, num_calls = self._verify([
        _verdict(False),
        _verdict(True),
        RuntimeError('model error'),
        _verdict(False),
    ])
    self.assertEqual(num_calls, 4)
    self.assertFalse(verdict.is_supported)
    self.assertEqual(info['recheck_votes_supported'], 1)
    self.assertEqual(info['recheck_votes_total'], 2)

    verdict, _, _ = self._verify(
        [
            _verdict(True),
            _verdict(False),
            RuntimeError('model error'),
            _verdict(True),
        ],
        claim_type='LATERALITY',
    )
    self.assertTrue(verdict.is_supported)

  def test_all_rechecks_failing_keeps_initial_verdict(self):
    initial = _verdict(False, correction='right knee')
    verdict, info, num_calls = self._verify(
        [initial] + [RuntimeError('model error')] * 3
    )
    self.assertEqual(num_calls, 4)
    self.assertEqual(verdict, initial)
    self.assertTrue(info['rechecked'])
    self.assertEqual(info['recheck_votes_total'], 0)

  def test_initial_pass_failure_propagates(self):
    with self.assertRaisesRegex(RuntimeError, 'model error'):
      self._verify([RuntimeError('model error')])

  def test_zero_recheck_runs_disables_rechecks(self):
    for claim_type in ('ACTION', 'LATERALITY'):
      verdict, info, num_calls = self._verify(
          [_verdict(False)], claim_type=claim_type, runs=0
      )
      self.assertEqual(num_calls, 1)
      self.assertFalse(verdict.is_supported)
      self.assertFalse(info['rechecked'])

  def test_custom_recheck_runs(self):
    _, info, num_calls = self._verify(
        [_verdict(False)] + [_verdict(True)] * 5, runs=5
    )
    self.assertEqual(num_calls, 6)
    self.assertEqual(info['recheck_votes_total'], 5)

  def test_legacy_mode_does_not_recheck_failed_claims(self):
    verdict, info, num_calls = self._verify(
        [_verdict(False)], recheck_failed_claims=False
    )
    self.assertEqual(num_calls, 1)
    self.assertFalse(verdict.is_supported)
    self.assertFalse(info['rechecked'])

  def test_legacy_mode_uses_majority_over_all_laterality_runs(self):
    verdict, info, num_calls = self._verify(
        [_verdict(False), _verdict(True), _verdict(True)],
        claim_type='LATERALITY',
        recheck_failed_claims=False,
    )
    self.assertEqual(num_calls, 3)
    self.assertTrue(verdict.is_supported)
    self.assertStartsWith(verdict.evidence, '[Self-consistency 2/3]')
    self.assertFalse(info['initial_is_supported'])
    self.assertTrue(info['rechecked'])
    self.assertEqual(info['recheck_votes_supported'], 2)
    self.assertEqual(info['recheck_votes_total'], 3)

  def test_legacy_mode_tie_is_broken_by_first_run(self):
    verdict, _, num_calls = self._verify(
        [_verdict(False), RuntimeError('model error'), _verdict(True)],
        claim_type='LATERALITY',
        recheck_failed_claims=False,
    )
    self.assertEqual(num_calls, 3)
    self.assertFalse(verdict.is_supported)

  def test_legacy_mode_raises_when_all_laterality_runs_fail(self):
    with self.assertRaisesRegex(RuntimeError, 'All 3 laterality runs failed'):
      self._verify(
          [RuntimeError('model error')] * 3,
          claim_type='LATERALITY',
          recheck_failed_claims=False,
      )

  def test_legacy_mode_single_run_for_laterality_when_runs_is_one(self):
    _, info, num_calls = self._verify(
        [_verdict(False)],
        claim_type='LATERALITY',
        recheck_failed_claims=False,
        runs=1,
    )
    self.assertEqual(num_calls, 1)
    self.assertFalse(info['rechecked'])


class FabricationHeuristicTest(absltest.TestCase):

  def _verify(
      self,
      side_effect,
      claim_type='ACTION',
      fabrication_keyword_heuristic=True,
      **config_kwargs,
  ):
    config = stage1b_verify.RecheckConfig(
        fabrication_keyword_heuristic=fabrication_keyword_heuristic,
        **config_kwargs,
    )
    with mock.patch.object(
        stage1b_verify, '_verify_claim', side_effect=side_effect
    ):
      verdict, _ = stage1b_verify._verify_claim_with_rechecks(
          claim_text=_CLAIM,
          claim_type=claim_type,
          clip_part=_CLIP_PART,
          lm=mock.MagicMock(),
          config=config,
      )
    return verdict

  def _fabricated_runs(self, num_runs):
    return [
        _verdict(
            False,
            evidence='The patient does not touch either knee.',
            correction='Patient rests both hands on the lap.',
        )
    ] * num_runs

  def test_heuristic_is_off_by_default(self):
    self.assertFalse(
        stage1b_verify.RecheckConfig().fabrication_keyword_heuristic
    )
    with mock.patch.object(
        stage1b_verify, '_verify_claim', side_effect=self._fabricated_runs(4)
    ):
      verdict, _ = stage1b_verify._verify_claim_with_rechecks(
          claim_text=_CLAIM,
          claim_type='ACTION',
          clip_part=_CLIP_PART,
          lm=mock.MagicMock(),
          config=stage1b_verify.RecheckConfig(),
      )
    self.assertEqual(
        verdict.suggested_correction, 'Patient rests both hands on the lap.'
    )

  def test_clears_correction_of_rechecked_fabricated_claim(self):
    verdict = self._verify(self._fabricated_runs(4))
    self.assertFalse(verdict.is_supported)
    self.assertEqual(verdict.suggested_correction, '')

  def test_disabled_heuristic_keeps_correction(self):
    verdict = self._verify(
        self._fabricated_runs(4), fabrication_keyword_heuristic=False
    )
    self.assertEqual(
        verdict.suggested_correction, 'Patient rests both hands on the lap.'
    )

  def test_wrong_detail_keeps_correction(self):
    verdict = self._verify(
        [
            _verdict(
                False, evidence='The right knee is shown.', correction='R'
            ),
        ]
        * 4
    )
    self.assertEqual(verdict.suggested_correction, 'R')

  def test_not_applied_without_rechecks(self):
    verdict = self._verify(self._fabricated_runs(1), runs=0)
    self.assertEqual(
        verdict.suggested_correction, 'Patient rests both hands on the lap.'
    )

  def test_legacy_mode_applies_only_to_laterality_runs(self):
    single_pass = self._verify(
        self._fabricated_runs(1), recheck_failed_claims=False
    )
    self.assertEqual(
        single_pass.suggested_correction,
        'Patient rests both hands on the lap.',
    )
    laterality = self._verify(
        self._fabricated_runs(3),
        claim_type='LATERALITY',
        recheck_failed_claims=False,
    )
    self.assertEqual(laterality.suggested_correction, '')

  def test_is_fabricated_keywords(self):
    self.assertTrue(stage1b_verify._is_fabricated('Hands are NOT VISIBLE.'))
    self.assertTrue(stage1b_verify._is_fabricated('No hand movement at all.'))
    self.assertFalse(
        stage1b_verify._is_fabricated('Patient touches the right knee.')
    )


class ResolveRecheckRunsTest(absltest.TestCase):

  def test_defaults_to_recheck_runs(self):
    self.assertEqual(
        stage1b_verify._resolve_recheck_runs(3, False, None, None), 3
    )

  def test_deprecated_aliases_override_default(self):
    self.assertEqual(stage1b_verify._resolve_recheck_runs(3, False, 5, None), 5)
    self.assertEqual(stage1b_verify._resolve_recheck_runs(3, False, None, 1), 1)

  def test_consistency_runs_takes_precedence(self):
    self.assertEqual(stage1b_verify._resolve_recheck_runs(3, False, 5, 7), 7)

  def test_matching_explicit_value_is_allowed(self):
    self.assertEqual(stage1b_verify._resolve_recheck_runs(5, True, 5, None), 5)

  def test_conflicting_explicit_value_raises(self):
    with self.assertRaisesRegex(ValueError, 'conflicts'):
      stage1b_verify._resolve_recheck_runs(3, True, 5, None)

  def test_negative_runs_raise(self):
    with self.assertRaisesRegex(ValueError, '>= 0'):
      stage1b_verify._resolve_recheck_runs(-1, True, None, None)


class ProcessItemTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    clips_dir = self.create_tempdir()
    clips_dir.create_file('clip_01.mp4', content='av-clip')
    self.clips_dir = clips_dir.full_path

  def test_records_recheck_fields_and_counts(self):
    item = {
        'clip': 'clip_01.mp4',
        'condition': 'Knee osteoarthritis',
        'symptom': 'Knee pain',
        'evidence': {'action': 'Patient points to the left knee.'},
    }
    claims = [
        stage1b_verify.AtomicClaim(
            claim_text='Patient points.', claim_type='ACTION'
        ),
        stage1b_verify.AtomicClaim(
            claim_text='The left knee.', claim_type='LATERALITY'
        ),
    ]
    verdicts = [
        _verdict(True),  # ACTION: initial pass only.
        _verdict(True),  # LATERALITY: initial pass ...
        _verdict(False, correction='The right knee.'),  # ... and 3 rechecks.
        _verdict(False, correction='The right knee.'),
        _verdict(True),
    ]
    with mock.patch.object(
        stage1b_verify, '_decompose_observation', return_value=claims
    ), mock.patch.object(
        stage1b_verify, '_verify_claim', side_effect=verdicts
    ) as mock_verify:
      result = stage1b_verify._process_item(
          item,
          lm=mock.MagicMock(),
          clips_dir=self.clips_dir,
          recheck_config=stage1b_verify.RecheckConfig(),
      )

    self.assertEqual(mock_verify.call_count, 5)
    verification = result['verification']
    self.assertEqual(verification['num_claims'], 2)
    self.assertEqual(verification['num_flagged'], 1)
    self.assertEqual(verification['num_rechecked'], 1)
    self.assertEqual(verification['num_overturned'], 1)
    self.assertTrue(verification['needs_review'])
    action, laterality = verification['all_claims']
    self.assertFalse(action['rechecked'])
    self.assertTrue(action['initial_is_supported'])
    self.assertEqual(laterality['claim_type'], 'LATERALITY')
    self.assertFalse(laterality['is_supported'])
    self.assertTrue(laterality['initial_is_supported'])
    self.assertEqual(laterality['recheck_votes_supported'], 1)
    self.assertEqual(laterality['recheck_votes_total'], 3)
    self.assertEqual(laterality['suggested_correction'], 'The right knee.')
    self.assertEqual(verification['flagged_claims'], [laterality])

  def test_missing_clip_raises_before_any_model_call(self):
    item = {
        'clip': 'missing_clip.mp4',
        'evidence': {'action': 'Patient points to the left knee.'},
    }
    lm = mock.MagicMock()
    with mock.patch.object(
        stage1b_verify, '_decompose_observation'
    ) as mock_decompose:
      with self.assertRaises(FileNotFoundError):
        stage1b_verify._process_item(
            item,
            lm=lm,
            clips_dir=self.clips_dir,
            recheck_config=stage1b_verify.RecheckConfig(),
        )

    mock_decompose.assert_not_called()
    lm.generate_thought_and_object.assert_not_called()

  def test_failed_claim_verification_fails_the_item(self):
    item = {
        'clip': 'clip_01.mp4',
        'evidence': {'action': 'Patient points to the left knee.'},
    }
    claims = [
        stage1b_verify.AtomicClaim(
            claim_text='Patient points.', claim_type='ACTION'
        ),
        stage1b_verify.AtomicClaim(
            claim_text='At the knee.', claim_type='ACTION'
        ),
    ]
    with mock.patch.object(
        stage1b_verify, '_decompose_observation', return_value=claims
    ), mock.patch.object(
        stage1b_verify,
        '_verify_claim',
        side_effect=[_verdict(True), RuntimeError('Quota exceeded')],
    ):
      with self.assertRaisesRegex(RuntimeError, 'Could not verify claim'):
        stage1b_verify._process_item(
            item,
            lm=mock.MagicMock(),
            clips_dir=self.clips_dir,
            recheck_config=stage1b_verify.RecheckConfig(),
        )

  def test_failed_decomposition_fails_the_item(self):
    item = {
        'clip': 'clip_01.mp4',
        'evidence': {'action': 'Patient points to the left knee.'},
    }
    with mock.patch.object(
        stage1b_verify,
        '_decompose_observation',
        side_effect=RuntimeError('Unparseable response'),
    ), mock.patch.object(stage1b_verify, '_verify_claim') as mock_verify:
      with self.assertRaisesRegex(RuntimeError, 'Could not decompose'):
        stage1b_verify._process_item(
            item,
            lm=mock.MagicMock(),
            clips_dir=self.clips_dir,
            recheck_config=stage1b_verify.RecheckConfig(),
        )
    mock_verify.assert_not_called()

  def test_failed_item_is_not_saved_so_resuming_retries_it(self):
    output_file = os.path.join(self.create_tempdir().full_path, 'stage1b.json')
    item = {'clip': 'clip_01.mp4', 'evidence': {}}
    with mock.patch.object(
        stage1b_verify,
        '_process_item',
        side_effect=RuntimeError('Could not verify claim'),
    ):
      verified = stage1b_verify._run_verification_pipeline(
          remaining_items=[item],
          verified_items=[],
          lm=mock.MagicMock(),
          clips_dir=self.clips_dir,
          output_file=output_file,
          recheck_config=stage1b_verify.RecheckConfig(),
          num_workers=1,
          model_name='test-model',
      )
    self.assertEmpty(verified)
    self.assertFalse(os.path.exists(output_file))


if __name__ == '__main__':
  absltest.main()
