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
"""Stage 1B: Automated GT verification via claim decomposition + audio.

Reads Stage 1 output, decomposes each EIM facet observation into atomic
claims, generates targeted verification probes, and runs them against
video+audio with constitutional camera-perspective rules.

Each claim receives one initial verification pass. Claims that fail this
initial check, as well as all LATERALITY claims (left/right confusions are a
frequent hallucination mode under ambiguous camera perspective), receive
--recheck_runs (default 3) additional independent checks, and the final
verdict is the majority vote across those rechecks. With
--norecheck_failed_claims, the legacy behavior used to build the released
benchmarks is reproduced instead: only LATERALITY claims are verified with
multiple self-consistency runs.

Outputs a verified Stage 1 file with flags and corrections.
"""

import collections
from concurrent import futures
import dataclasses
import json
import os
import threading
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import pydantic

import constraints
import content_utils
import language_model
import template_utils

_STAGE1_FILE = flags.DEFINE_string(
    'stage1_file', None, 'Path to Stage 1 output JSON.'
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file', None, 'Path to write verified output.'
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', language_model.DEFAULT_MODEL, 'Gemini model ID.'
)
_WORKERS = flags.DEFINE_integer('workers', 10, 'Parallel workers.')
_CLIPS_DIR = flags.DEFINE_string(
    'clips_dir',
    None,
    'Directory containing the Stage 1 clips with audio (for Stage 0 clips, the'
    ' stage0_discover.py --output_clips_dir).',
)
_RECHECK_RUNS = flags.DEFINE_integer(
    'recheck_runs',
    3,
    'Number of additional independent verification runs for each rechecked'
    ' claim. The final verdict is the majority vote across these rechecks;'
    ' ties are broken by the initial verdict. 0 disables rechecks. With'
    ' --norecheck_failed_claims, this is the total number of'
    ' self-consistency runs for LATERALITY claims.',
)
_RECHECK_FAILED_CLAIMS = flags.DEFINE_bool(
    'recheck_failed_claims',
    True,
    'If True (the method described in the paper), every claim gets one'
    ' initial verification pass, and claims that fail it, as well as all'
    ' LATERALITY claims, get --recheck_runs additional checks decided by'
    ' majority vote. If False, reproduces the legacy behavior used to build'
    ' the released benchmarks: LATERALITY claims are verified with'
    ' --recheck_runs self-consistency runs in total (majority vote, no'
    ' separate initial pass) and all other claims get a single pass.',
)
_LATERALITY_RUNS = flags.DEFINE_integer(
    'laterality_runs',
    None,
    'DEPRECATED: use --recheck_runs. If set, overrides --recheck_runs.',
)
_CONSISTENCY_RUNS = flags.DEFINE_integer(
    'consistency_runs',
    None,
    'DEPRECATED: use --recheck_runs. If set, overrides --recheck_runs and'
    ' --laterality_runs.',
)
_FABRICATION_KEYWORD_HEURISTIC = flags.DEFINE_bool(
    'fabrication_keyword_heuristic',
    False,
    'Optional variant from a later revision of our internal pipeline; it was'
    ' not used to build the released benchmarks. If True, a rechecked claim'
    ' that remains unsupported is treated as fabricated when the consensus'
    ' evidence contains phrases such as "not visible" or "does not"; its'
    ' suggested correction is then cleared so that Stage 1B-patch removes the'
    ' claim instead of rewriting it. This is a keyword heuristic with known'
    ' false positives (e.g. "does not" also matches evidence that only'
    ' disputes a detail). With --norecheck_failed_claims it applies only to'
    ' multi-run LATERALITY claims.',
)

_LOCK = threading.Lock()

# Evidence phrases suggesting that the claimed action or event is not visible
# at all, rather than described with a wrong detail.
_FABRICATION_SIGNALS = (
    'not visible',
    'not seen',
    'not observed',
    'no hand',
    'no movement',
    'does not',
    'do not',
    'is not visible',
    'is not seen',
    'is not present',
    'are not visible',
    'are not seen',
    'cannot be seen',
    'not present',
    'hands are not',
    'not in frame',
    'not appear',
)


@dataclasses.dataclass(frozen=True)
class RecheckConfig:
  """How claims are re-verified; see the --recheck_* flags."""

  runs: int = 3
  recheck_failed_claims: bool = True
  fabrication_keyword_heuristic: bool = False


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class AtomicClaim(constraints.BaseModel):
  """A single atomic claim extracted from an observation."""

  claim_text: str = pydantic.Field(
      description=(
          'A single, specific, independently verifiable factual claim'
          ' extracted from the observation.'
      )
  )
  claim_type: str = pydantic.Field(
      description=(
          'One of: LATERALITY, ACTION, ANATOMY, TEMPORAL, APPEARANCE, ABSENCE,'
          ' OTHER'
      )
  )


@pydantic.dataclasses.dataclass
class ClaimList(constraints.BaseModel):
  """List of atomic claims from an observation."""

  claims: list[AtomicClaim] = pydantic.Field(
      description='All atomic claims extracted.'
  )


@pydantic.dataclasses.dataclass
class ClaimVerdict(constraints.BaseModel):
  """Verification result for a single claim."""

  is_supported: bool = pydantic.Field(
      description=(
          'True if the claim is clearly supported by the video and audio'
          ' evidence.'
      )
  )
  confidence: str = pydantic.Field(
      description='HIGH, MEDIUM, or LOW confidence.'
  )
  evidence: str = pydantic.Field(
      description=(
          'What you see and hear that supports or contradicts this claim. Be'
          ' specific.'
      )
  )
  suggested_correction: str = pydantic.Field(
      description=(
          'If is_supported is False, suggest the correct version. Otherwise'
          ' empty string.'
      )
  )


# ---------------------------------------------------------------------------
# Verification routines
# ---------------------------------------------------------------------------


def _decompose_observation(
    observation: str,
    clip: str,
    condition: str,
    symptom: str,
    lm: language_model.LanguageModel,
) -> list[AtomicClaim]:
  """Decomposes an observation string into atomic claims."""
  prompt = template_utils.render_template(
      'stage1b_decompose_prompt',
      observation=observation,
      clip=clip,
      condition=condition,
      symptom=symptom,
  )
  prompt += constraints.get_constraint_suffix(cls=ClaimList)
  result, _ = lm.generate_thought_and_object(prompt, cls=ClaimList)
  return result.claims


def _verify_claim(
    claim_text: str,
    clip_part: dict[str, Any],
    lm: language_model.LanguageModel,
) -> ClaimVerdict:
  """Verifies a single claim against video+audio.

  Args:
    claim_text: The claim to verify.
    clip_part: The clip (with audio) as a part from content_utils.media_part.
    lm: Language model instance.

  Returns:
    The model's verdict.
  """
  prompt = template_utils.render_template(
      'stage1b_verify_claim_prompt',
      claim=claim_text,
  )
  prompt += constraints.get_constraint_suffix(cls=ClaimVerdict)
  result, _ = lm.generate_thought_and_object(
      [clip_part, prompt], cls=ClaimVerdict
  )
  return result


def _is_fabricated(evidence: str) -> bool:
  """Returns True if verifier evidence suggests the whole event is absent.

  This is a keyword heuristic with known false positives: e.g. "does not"
  also matches evidence that disputes only a detail of an event that did
  happen. An LLM judgment of whether the claimed event is absent from the clip
  would be more robust.

  Args:
    evidence: Evidence text of an unsupported verdict.
  """
  lowered = evidence.lower()
  return any(signal in lowered for signal in _FABRICATION_SIGNALS)


def _reach_claim_verdict(
    verdicts: Sequence[ClaimVerdict],
    claim_text: str,
    tie_breaker: bool | None = None,
    apply_fabrication_heuristic: bool = False,
    vote_label: str = 'Self-consistency',
) -> ClaimVerdict:
  """Aggregates multiple verification runs into a consensus ClaimVerdict.

  Args:
    verdicts: Successful verification runs (at least one).
    claim_text: The claim being verified (used for logging).
    tie_breaker: Verdict to use when supported and unsupported votes are tied.
      If None, the first run's verdict wins (legacy behavior).
    apply_fabrication_heuristic: If True and the consensus is unsupported,
      clears the suggested correction when the evidence suggests that the
      claimed event is not visible at all (see _is_fabricated), so that Stage
      1B-patch removes the claim instead of rewriting it.
    vote_label: Label used in the evidence prefix and in logs.

  Returns:
    The consensus ClaimVerdict.
  """
  num_supported = sum(1 for v in verdicts if v.is_supported)
  num_unsupported = len(verdicts) - num_supported
  if num_supported != num_unsupported:
    majority_supported = num_supported > num_unsupported
  elif tie_breaker is not None:
    majority_supported = tie_breaker
  else:
    majority_supported = verdicts[0].is_supported
  agreement = num_supported if majority_supported else num_unsupported
  total = len(verdicts)

  logging.info(
      '  %s: "%s" → %s (%d/%d agree)',
      vote_label,
      claim_text[:60],
      'SUPPORTED' if majority_supported else 'FLAGGED',
      agreement,
      total,
  )

  if majority_supported:
    supported = [v for v in verdicts if v.is_supported]
    best = max(
        supported,
        key=lambda v: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(v.confidence, 0),
    )
    return ClaimVerdict(
        is_supported=True,
        confidence=best.confidence if agreement == total else 'MEDIUM',
        evidence=(f'[{vote_label} {agreement}/{total}] ' + best.evidence),
        suggested_correction='',
    )

  unsupported = [v for v in verdicts if not v.is_supported]
  corrections = [
      v.suggested_correction for v in unsupported if v.suggested_correction
  ]
  best_correction = (
      collections.Counter(corrections).most_common(1)[0][0]
      if corrections
      else ''
  )
  best = unsupported[0]
  if apply_fabrication_heuristic and _is_fabricated(best.evidence):
    logging.info(
        '  Fabricated claim detected; clearing correction: "%s"',
        claim_text[:60],
    )
    best_correction = ''
  return ClaimVerdict(
      is_supported=False,
      confidence=best.confidence if agreement == total else 'MEDIUM',
      evidence=(f'[{vote_label} {agreement}/{total}] ' + best.evidence),
      suggested_correction=best_correction,
  )


def _run_verification_passes(
    claim_text: str,
    clip_part: dict[str, Any],
    lm: language_model.LanguageModel,
    num_runs: int,
    run_label: str,
) -> list[ClaimVerdict]:
  """Runs independent verification passes, skipping the ones that fail."""
  verdicts: list[ClaimVerdict] = []
  for run_idx in range(num_runs):
    try:
      verdicts.append(_verify_claim(claim_text, clip_part, lm))
    except Exception as e:  # pylint: disable=broad-except
      logging.warning(
          '%s run %d/%d failed for "%s": %s',
          run_label,
          run_idx + 1,
          num_runs,
          claim_text[:50],
          e,
      )
  return verdicts


def _recheck_info(
    initial_is_supported: bool,
    rechecked: bool,
    votes: Sequence[ClaimVerdict],
) -> dict[str, Any]:
  """Per-claim recheck fields recorded in the Stage 1B output."""
  return {
      'initial_is_supported': initial_is_supported,
      'rechecked': rechecked,
      'recheck_votes_supported': sum(1 for v in votes if v.is_supported),
      'recheck_votes_total': len(votes),
  }


def _verify_claim_with_rechecks(
    claim_text: str,
    claim_type: str,
    clip_part: dict[str, Any],
    lm: language_model.LanguageModel,
    config: RecheckConfig,
) -> tuple[ClaimVerdict, dict[str, Any]]:
  """Verifies a claim according to the recheck policy.

  By default, the claim gets one initial verification pass. If it fails that
  pass, or if it is a LATERALITY claim (spatial laterality, e.g. left vs.
  right arm/chest, is a frequent hallucination mode under ambiguous camera
  perspective), config.runs additional independent checks are run and the
  majority vote across those rechecks decides; ties are broken by the initial
  verdict, and the initial verdict is kept if every recheck fails.

  With config.recheck_failed_claims=False (legacy), only LATERALITY claims
  get multiple runs: config.runs self-consistency runs in total, decided by
  majority vote. Their first run is reported as the initial verdict.

  Args:
    claim_text: The claim to verify.
    claim_type: The claim type from decomposition, e.g. 'LATERALITY'.
    clip_part: The clip (with audio) as a part from content_utils.media_part.
    lm: Language model instance.
    config: Recheck policy.

  Returns:
    The final verdict and the claim's recheck fields (initial_is_supported,
    rechecked, recheck_votes_supported, recheck_votes_total).

  Raises:
    Exception: If the initial pass fails or, in legacy mode, if every
      LATERALITY run fails.
  """
  is_laterality = claim_type == 'LATERALITY'

  if not config.recheck_failed_claims:
    if is_laterality and config.runs > 1:
      verdicts = _run_verification_passes(
          claim_text, clip_part, lm, config.runs, 'Laterality'
      )
      if not verdicts:
        raise RuntimeError(
            f'All {config.runs} laterality runs failed for: {claim_text}'
        )
      verdict = _reach_claim_verdict(
          verdicts,
          claim_text,
          apply_fabrication_heuristic=config.fabrication_keyword_heuristic,
      )
      return verdict, _recheck_info(verdicts[0].is_supported, True, verdicts)
    verdict = _verify_claim(claim_text, clip_part, lm)
    return verdict, _recheck_info(verdict.is_supported, False, [])

  initial = _verify_claim(claim_text, clip_part, lm)
  if config.runs <= 0 or (initial.is_supported and not is_laterality):
    return initial, _recheck_info(initial.is_supported, False, [])

  rechecks = _run_verification_passes(
      claim_text, clip_part, lm, config.runs, 'Recheck'
  )
  if not rechecks:
    logging.warning(
        'All %d rechecks failed for "%s"; keeping the initial verdict.',
        config.runs,
        claim_text[:50],
    )
    return initial, _recheck_info(initial.is_supported, True, [])
  verdict = _reach_claim_verdict(
      rechecks,
      claim_text,
      tie_breaker=initial.is_supported,
      apply_fabrication_heuristic=config.fabrication_keyword_heuristic,
      vote_label='Recheck majority',
  )
  return verdict, _recheck_info(initial.is_supported, True, rechecks)


def _process_item(
    item: dict[str, Any],
    lm: language_model.LanguageModel,
    clips_dir: str,
    recheck_config: RecheckConfig,
) -> dict[str, Any]:
  """Processes a single Stage 1 item through claim decomposition and verification.

  Args:
    item: Stage 1 result for one clip.
    lm: Language model instance.
    clips_dir: Directory containing the clip (with audio) named item['clip'].
    recheck_config: Recheck policy.

  Returns:
    A copy of the item with a 'verification' entry.

  Raises:
    FileNotFoundError: If the clip does not exist.
    RuntimeError: If the evidence of a facet could not be decomposed into
      claims or a claim could not be verified (e.g. because of API errors).
      The item is then left out of the output, so that it is retried when the
      run is resumed, rather than recorded with unverified claims.
  """
  clip = item['clip']
  clip_path = os.path.join(clips_dir, clip)
  if not os.path.exists(clip_path):
    raise FileNotFoundError(f'Clip not found: {clip_path}')
  condition = item.get('condition', '')
  symptom = item.get('symptom', '')
  evidence = item.get('evidence', {})

  logging.info('Processing %s (%s)', clip, condition)

  facet_texts = {
      facet: evidence[facet]
      for facet in [
          'action',
          'spatial',
          'temporal',
          'appearance',
          'absence',
          'primary_sign',
      ]
      if facet in evidence and evidence[facet]
  }

  all_claims = []
  for facet, text in facet_texts.items():
    try:
      claims = _decompose_observation(
          observation=text,
          clip=clip,
          condition=condition,
          symptom=symptom,
          lm=lm,
      )
    except Exception as e:
      raise RuntimeError(
          f'Could not decompose the {facet} evidence of {clip} into claims.'
      ) from e
    for c in claims:
      all_claims.append({
          'facet': facet,
          'claim': c.claim_text,
          'type': c.claim_type,
      })

  logging.info('  %s: %d claims extracted', clip, len(all_claims))

  claim_results = []
  if all_claims:
    # The clip is read (or uploaded, if large) once for all of its claims.
    with content_utils.media_part(lm.client, clip_path) as clip_part:
      for claim_info in all_claims:
        try:
          verdict, recheck_info = _verify_claim_with_rechecks(
              claim_text=claim_info['claim'],
              claim_type=claim_info['type'],
              clip_part=clip_part,
              lm=lm,
              config=recheck_config,
          )
        except Exception as e:
          raise RuntimeError(
              f'Could not verify claim "{claim_info["claim"][:50]}" of {clip}.'
          ) from e
        claim_results.append({
            'facet': claim_info['facet'],
            'claim': claim_info['claim'],
            'claim_type': claim_info['type'],
            'is_supported': verdict.is_supported,
            'confidence': verdict.confidence,
            'evidence': verdict.evidence,
            'suggested_correction': verdict.suggested_correction,
            **recheck_info,
        })
  flagged_claims = [c for c in claim_results if not c['is_supported']]

  n_flagged = len(flagged_claims)
  n_total = len(claim_results)
  n_rechecked = sum(1 for c in claim_results if c['rechecked'])
  n_overturned = sum(
      1
      for c in claim_results
      if c['rechecked'] and c['is_supported'] != c['initial_is_supported']
  )
  logging.info(
      '  %s: %d/%d claims flagged (%d rechecked, %d overturned)',
      clip,
      n_flagged,
      n_total,
      n_rechecked,
      n_overturned,
  )

  verified = dict(item)
  verified['verification'] = {
      'num_claims': n_total,
      'num_flagged': n_flagged,
      'flagged_ratio': n_flagged / n_total if n_total else 0.0,
      'num_rechecked': n_rechecked,
      'num_overturned': n_overturned,
      'all_claims': claim_results,
      'flagged_claims': flagged_claims,
      'needs_review': n_flagged > 0,
  }
  return verified


# ---------------------------------------------------------------------------
# Pipeline orchestration & summaries
# ---------------------------------------------------------------------------


def _load_stage1_data(
    stage1_file: str,
    output_file: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], str]:
  """Loads Stage 1 input items and any existing checkpoint."""
  with open(stage1_file, 'r') as f:
    data = json.load(f)

  items = data.get('results', [])
  logging.info('Loaded %d Stage 1 items.', len(items))

  verified_items = []
  done_clips = set()
  if os.path.exists(output_file):
    try:
      with open(output_file, 'r') as f:
        checkpoint = json.load(f)
      verified_items = checkpoint.get('results', [])
      done_clips = {v['clip'] for v in verified_items if 'clip' in v}
      logging.info('Resumed from checkpoint: %d done.', len(done_clips))
    except (json.JSONDecodeError, KeyError):
      pass

  return items, verified_items, done_clips, data.get('model', '')


def _run_verification_pipeline(
    remaining_items: list[dict[str, Any]],
    verified_items: list[dict[str, Any]],
    lm: language_model.LanguageModel,
    clips_dir: str,
    output_file: str,
    recheck_config: RecheckConfig,
    num_workers: int,
    model_name: str,
) -> list[dict[str, Any]]:
  """Executes multithreaded verification across remaining items."""

  def _process_and_save(item: dict[str, Any]) -> dict[str, Any]:
    result = _process_item(item, lm, clips_dir, recheck_config)
    with _LOCK:
      verified_items.append(result)
      output = {
          'model': model_name,
          'pipeline': 'stage1b_verification',
          'recheck_runs': recheck_config.runs,
          'recheck_failed_claims': recheck_config.recheck_failed_claims,
          'fabrication_keyword_heuristic': (
              recheck_config.fabrication_keyword_heuristic
          ),
          'num_processed': len(verified_items),
          'num_flagged': sum(
              1
              for v in verified_items
              if v.get('verification', {}).get('needs_review')
          ),
          'results': verified_items,
      }
      with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    return result

  with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futs = {
        executor.submit(_process_and_save, item): item
        for item in remaining_items
    }
    for fut in futures.as_completed(futs):
      item = futs[fut]
      try:
        result = fut.result()
        v = result.get('verification', {})
        status = 'FLAGGED' if v.get('needs_review') else 'OK'
        logging.info(
            '  %s [%s] — %d/%d claims flagged',
            result['clip'],
            status,
            v.get('num_flagged', 0),
            v.get('num_claims', 0),
        )
      except Exception as e:  # pylint: disable=broad-except
        logging.error('Failed: %s — %s', item['clip'], e)

  return verified_items


def _print_verification_summary(
    verified_items: list[dict[str, Any]],
    recheck_config: RecheckConfig,
) -> None:
  """Prints final summary metrics for verified claims and flag rates."""
  total = len(verified_items)
  flagged = sum(
      1 for v in verified_items if v.get('verification', {}).get('needs_review')
  )
  logging.info('=' * 60)
  logging.info('STAGE 1B VERIFICATION COMPLETE')
  logging.info('  Total items: %d', total)
  logging.info(
      '  Flagged for review: %d (%.1f%%)',
      flagged,
      flagged / total * 100 if total else 0,
  )
  logging.info(
      '  Clean items: %d (%.1f%%)',
      total - flagged,
      (total - flagged) / total * 100 if total else 0,
  )
  if recheck_config.recheck_failed_claims:
    logging.info(
        '  Recheck policy: initial pass + %d rechecks for failed and'
        ' LATERALITY claims',
        recheck_config.runs,
    )
  else:
    logging.info(
        '  Recheck policy (legacy): %d self-consistency runs for LATERALITY'
        ' claims only',
        recheck_config.runs,
    )

  facet_flags: collections.Counter[str] = collections.Counter()
  facet_totals: collections.Counter[str] = collections.Counter()
  num_rechecked = 0
  num_to_supported = 0
  num_to_unsupported = 0
  for v in verified_items:
    for c in v.get('verification', {}).get('all_claims', []):
      claim_type = c.get('claim_type', 'OTHER')
      facet_totals[claim_type] += 1
      if not c.get('is_supported'):
        facet_flags[claim_type] += 1
      if c.get('rechecked'):
        num_rechecked += 1
        if c.get('is_supported') and not c.get('initial_is_supported'):
          num_to_supported += 1
        elif not c.get('is_supported') and c.get('initial_is_supported'):
          num_to_unsupported += 1

  num_claims = sum(facet_totals.values())
  logging.info(
      '  Rechecked claims: %d/%d (%.1f%%)',
      num_rechecked,
      num_claims,
      num_rechecked / num_claims * 100 if num_claims else 0,
  )
  logging.info(
      '  Overturned verdicts: %d (%d now supported, %d now unsupported)',
      num_to_supported + num_to_unsupported,
      num_to_supported,
      num_to_unsupported,
  )

  logging.info('  Per claim-type flag rates:')
  for ct in sorted(facet_totals.keys()):
    f = facet_flags[ct]
    t = facet_totals[ct]
    logging.info('    %s: %d/%d (%.0f%%)', ct, f, t, f / t * 100 if t else 0)


def _resolve_recheck_runs(
    recheck_runs: int,
    recheck_runs_present: bool,
    laterality_runs: int | None,
    consistency_runs: int | None,
) -> int:
  """Resolves --recheck_runs and its deprecated aliases.

  Args:
    recheck_runs: Value of --recheck_runs.
    recheck_runs_present: Whether --recheck_runs was set explicitly.
    laterality_runs: Value of the deprecated --laterality_runs, or None.
    consistency_runs: Value of the deprecated --consistency_runs, or None. Takes
      precedence over laterality_runs.

  Returns:
    The number of recheck runs to use.

  Raises:
    ValueError: If an alias conflicts with an explicit --recheck_runs, or the
      resolved value is negative.
  """
  alias_name, alias_value = 'consistency_runs', consistency_runs
  if alias_value is None:
    alias_name, alias_value = 'laterality_runs', laterality_runs
  if alias_value is None:
    runs = recheck_runs
  elif recheck_runs_present and alias_value != recheck_runs:
    raise ValueError(
        f'--{alias_name}={alias_value} conflicts with'
        f' --recheck_runs={recheck_runs}; set only --recheck_runs.'
    )
  else:
    logging.warning(
        '--%s is deprecated; use --recheck_runs instead.', alias_name
    )
    runs = alias_value
  if runs < 0:
    raise ValueError(f'--recheck_runs must be >= 0, got {runs}.')
  return runs


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> None:
  """Runs Stage 1B claim decomposition and verification pipeline."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  assert _STAGE1_FILE.value is not None
  assert _OUTPUT_FILE.value is not None
  assert _CLIPS_DIR.value is not None

  try:
    recheck_runs = _resolve_recheck_runs(
        recheck_runs=_RECHECK_RUNS.value,
        recheck_runs_present=_RECHECK_RUNS.present,
        laterality_runs=_LATERALITY_RUNS.value,
        consistency_runs=_CONSISTENCY_RUNS.value,
    )
  except ValueError as e:
    raise app.UsageError(str(e)) from e
  recheck_config = RecheckConfig(
      runs=recheck_runs,
      recheck_failed_claims=_RECHECK_FAILED_CLAIMS.value,
      fabrication_keyword_heuristic=_FABRICATION_KEYWORD_HEURISTIC.value,
  )

  items, verified_items, done_clips, orig_model = _load_stage1_data(
      stage1_file=_STAGE1_FILE.value,
      output_file=_OUTPUT_FILE.value,
  )

  remaining = [i for i in items if i['clip'] not in done_clips]
  logging.info('%d items remaining to verify.', len(remaining))

  lm = language_model.LanguageModel(model_name=_MODEL_NAME.value)

  verified_results = _run_verification_pipeline(
      remaining_items=remaining,
      verified_items=verified_items,
      lm=lm,
      clips_dir=_CLIPS_DIR.value,
      output_file=_OUTPUT_FILE.value,
      recheck_config=recheck_config,
      num_workers=_WORKERS.value,
      model_name=orig_model or _MODEL_NAME.value,
  )

  _print_verification_summary(verified_results, recheck_config)


if __name__ == '__main__':
  flags.mark_flags_as_required(['stage1_file', 'output_file', 'clips_dir'])
  app.run(main)
