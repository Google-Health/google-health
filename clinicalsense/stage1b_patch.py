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
"""Stage 1B-patch: Apply verified corrections to Stage 1 evidence.

Pipeline execution order:
  Stage 1 (prototype extraction)
    -> Stage 1B (verify claims against video/audio)
    -> Stage 1B-patch (apply verified corrections, THIS SCRIPT)
    -> Stage 2 (MCQ question generation)

Reads Stage 1B verification output produced by stage1b_verify.py,
identifies flagged claims, and surgically edits the Stage 1 evidence fields
using Jinja2 prompts + LLM editing:
  - Flagged claims with a suggested correction (and confidence above LOW)
    have the incorrect text replaced by the correction.
  - Flagged claims without a suggested correction (e.g. claims whose event
    the verifier did not see at all) are removed from the evidence entirely.
Produces a patched Stage 1 output ready for downstream Stage 2 generation.
"""

import json
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import pydantic

import constraints
import language_model
import template_utils

_STAGE1B_FILE = flags.DEFINE_string(
    'stage1b_file', None, 'Path to Stage 1B verification output.'
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file', None, 'Path to write patched Stage 1 output.'
)
_DIFF_FILE = flags.DEFINE_string(
    'diff_file',
    None,
    'Path to write human-readable diff report.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', language_model.DEFAULT_MODEL, 'Gemini model ID.'
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class PatchedEvidence(constraints.BaseModel):
  """A corrected version of a single evidence field."""

  corrected_text: str = pydantic.Field(
      description=(
          'The corrected evidence text with only the flagged claims fixed.'
          ' Keep all other content unchanged.'
      )
  )
  changes_made: list[str] = pydantic.Field(
      description='List of specific changes made.'
  )


# ---------------------------------------------------------------------------
# Patching routines
# ---------------------------------------------------------------------------


def _apply_patches_with_llm(
    original_text: str,
    facet: str,
    corrections: list[dict[str, Any]],
    lm: language_model.LanguageModel,
) -> tuple[str, list[str]]:
  """Applies verified corrections surgically using LLM."""
  corrections_text = '\n'.join(
      f'- WRONG: "{c["claim"]}"\n  CORRECT: "{c["suggested_correction"]}"'
      for c in corrections
  )
  prompt = template_utils.render_template(
      'stage1b_patch_prompt',
      facet=facet,
      original=original_text,
      corrections=corrections_text,
  )
  prompt += constraints.get_constraint_suffix(cls=PatchedEvidence)
  result, _ = lm.generate_thought_and_object(prompt, cls=PatchedEvidence)
  return result.corrected_text, result.changes_made


def _apply_patches_simple(
    original_text: str,
    corrections: list[dict[str, Any]],
) -> tuple[str, list[str]]:
  """Applies simple string replacement fallback for verified corrections."""
  patched = original_text
  changes = []
  for c in corrections:
    claim = c['claim']
    fix = c['suggested_correction']
    if not fix:
      continue
    if claim in patched:
      patched = patched.replace(claim, fix, 1)
      changes.append(f'Replaced: "{claim}" -> "{fix}"')
    else:
      changes.append(f'MANUAL: "{claim}" -> "{fix}" (not found verbatim)')
  return patched, changes


def _excise_claims_with_llm(
    original_text: str,
    facet: str,
    claims: list[str],
    lm: language_model.LanguageModel,
) -> tuple[str, list[str]]:
  """Removes fabricated claims from an evidence field using the LLM."""
  excisions_text = '\n'.join(
      f'- FABRICATED (remove entirely): "{claim}"' for claim in claims
  )
  prompt = template_utils.render_template(
      'stage1b_patch_prompt',
      facet=facet,
      original=original_text,
      corrections=excisions_text,
  )
  prompt += constraints.get_constraint_suffix(cls=PatchedEvidence)
  result, _ = lm.generate_thought_and_object(prompt, cls=PatchedEvidence)
  return result.corrected_text, result.changes_made


def _excise_claims_simple(
    original_text: str,
    claims: list[str],
) -> tuple[str, list[str]]:
  """Removes fabricated claims by exact substring match (fallback)."""
  patched = original_text
  changes = []
  for claim in claims:
    if claim in patched:
      patched = patched.replace(claim, '', 1).strip()
      changes.append(f'Removed: "{claim[:80]}"')
    else:
      changes.append(f'MANUAL: remove "{claim[:80]}" (not found verbatim)')
  return patched, changes


def _group_by_facet(
    claims: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
  """Groups flagged claims by the evidence facet they came from."""
  by_facet: dict[str, list[dict[str, Any]]] = {}
  for c in claims:
    by_facet.setdefault(c['facet'], []).append(c)
  return by_facet


def _patch_single_item(
    item: dict[str, Any],
    lm: language_model.LanguageModel,
) -> tuple[dict[str, Any], int, bool, list[str]]:
  """Applies facet corrections and excisions to a single item.

  Args:
    item: A Stage 1B verified item.
    lm: Language model used to edit the evidence text.

  Returns:
    (patched item, number of claims patched, whether the item was patched,
    diff lines for this item). The diff lines are built per item so that
    entries from different items never interleave.
  """
  item_diff: list[str] = []
  verification = item.get('verification', {})
  flagged = verification.get('flagged_claims', [])

  if not flagged:
    patched_item = dict(item)
    patched_item.pop('verification', None)
    return patched_item, 0, False, item_diff

  actionable = [
      c
      for c in flagged
      if c.get('suggested_correction') and c.get('confidence', 'HIGH') != 'LOW'
  ]
  # Flagged claims without a correction (the verifier found no valid
  # correction, or the optional Stage 1B fabrication heuristic cleared it) are
  # excised entirely.
  excisions = [
      c
      for c in flagged
      if not c.get('suggested_correction') and not c.get('is_supported', True)
  ]

  if not actionable and not excisions:
    patched_item = dict(item)
    patched_item.pop('verification', None)
    return patched_item, 0, False, item_diff

  clip = item['clip']
  evidence = dict(item.get('evidence', {}))
  item_diff.append(f'=== {clip} ({item.get("condition", "")}) ===')

  total_item_patches = 0
  # Phase 1: apply corrections (wrong details).
  for facet, corrections in _group_by_facet(actionable).items():
    if facet not in evidence:
      continue

    original = evidence[facet]
    item_diff.append(f'\n  FACET: {facet}')
    item_diff.append(f'  BEFORE: {original[:300]}')

    try:
      patched_text, changes = _apply_patches_with_llm(
          original, facet, corrections, lm
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('LLM patch failed for %s/%s: %s', clip, facet, e)
      patched_text, changes = _apply_patches_simple(original, corrections)
    evidence[facet] = patched_text
    total_item_patches += len(corrections)
    item_diff.append(f'  AFTER:  {patched_text[:300]}')
    for change in changes:
      item_diff.append(f'    • {change}')

  # Phase 2: excise claims that have no valid correction.
  for facet, to_excise in _group_by_facet(excisions).items():
    if facet not in evidence:
      continue

    original = evidence[facet]
    claims = [c['claim'] for c in to_excise]
    item_diff.append(f'\n  FACET: {facet} [EXCISION]')
    item_diff.append(f'  BEFORE: {original[:300]}')

    try:
      patched_text, changes = _excise_claims_with_llm(
          original, facet, claims, lm
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('LLM excision failed for %s/%s: %s', clip, facet, e)
      patched_text, changes = _excise_claims_simple(original, claims)
    evidence[facet] = patched_text
    total_item_patches += len(to_excise)
    item_diff.append(f'  AFTER:  {patched_text[:300]}')
    for change in changes:
      item_diff.append(f'    ✂ {change}')

  item_diff.append('')

  patched_item = dict(item)
  patched_item['evidence'] = evidence
  patched_item['patched'] = True
  patched_item['num_patches'] = total_item_patches
  patched_item.pop('verification', None)
  return patched_item, total_item_patches, True, item_diff


# ---------------------------------------------------------------------------
# Pipeline orchestration & summaries
# ---------------------------------------------------------------------------


def _load_verification_data(
    stage1b_file: str,
) -> tuple[list[dict[str, Any]], str]:
  """Loads Stage 1B verification JSON data."""
  with open(stage1b_file, 'r') as f:
    data = json.load(f)
  items = data.get('results', [])
  logging.info('Loaded %d verified items.', len(items))
  return items, data.get('model', '')


def _patch_all_items(
    items: list[dict[str, Any]],
    lm: language_model.LanguageModel,
) -> tuple[list[dict[str, Any]], list[str], int, int]:
  """Applies patches across all items and builds diff lines."""
  diff_lines: list[str] = []
  patched_items: list[dict[str, Any]] = []
  total_corrections = 0
  total_items_patched = 0

  for item in items:
    patched_item, n_patches, was_patched, item_diff = _patch_single_item(
        item, lm
    )
    diff_lines.extend(item_diff)
    patched_items.append(patched_item)
    total_corrections += n_patches
    if was_patched:
      total_items_patched += 1

  return patched_items, diff_lines, total_items_patched, total_corrections


def _save_patched_results(
    patched_items: list[dict[str, Any]],
    diff_lines: list[str],
    output_file: str,
    diff_file: str | None,
    model_name: str,
    total_items_patched: int,
    total_corrections: int,
) -> None:
  """Writes patched Stage 1 JSON and optional diff report."""
  output = {
      'model': model_name,
      'pipeline': 'stage1_patched',
      'num_processed': len(patched_items),
      'num_patched': total_items_patched,
      'total_corrections': total_corrections,
      'results': patched_items,
  }

  with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)
  logging.info('Wrote patched output to %s', output_file)

  if diff_file:
    with open(diff_file, 'w') as f:
      f.write('\n'.join(diff_lines))
    logging.info('Wrote diff report to %s', diff_file)


def _print_patch_summary(
    total_items: int,
    total_items_patched: int,
    total_corrections: int,
) -> None:
  """Logs summary metrics for the patching run."""
  logging.info('=' * 60)
  logging.info('STAGE 1B-PATCH COMPLETE')
  logging.info('  Total items: %d', total_items)
  logging.info('  Items patched: %d', total_items_patched)
  logging.info('  Total corrections applied: %d', total_corrections)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> None:
  """Runs Stage 1B-patch to apply verified corrections."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  assert _STAGE1B_FILE.value is not None
  assert _OUTPUT_FILE.value is not None

  items, orig_model = _load_verification_data(_STAGE1B_FILE.value)
  lm = language_model.LanguageModel(model_name=_MODEL_NAME.value)

  (
      patched_items,
      diff_lines,
      total_items_patched,
      total_corrections,
  ) = _patch_all_items(items, lm)

  _save_patched_results(
      patched_items=patched_items,
      diff_lines=diff_lines,
      output_file=_OUTPUT_FILE.value,
      diff_file=_DIFF_FILE.value,
      model_name=orig_model or _MODEL_NAME.value,
      total_items_patched=total_items_patched,
      total_corrections=total_corrections,
  )

  _print_patch_summary(
      len(patched_items), total_items_patched, total_corrections
  )


if __name__ == '__main__':
  flags.mark_flags_as_required(['stage1b_file', 'output_file'])
  app.run(main)
