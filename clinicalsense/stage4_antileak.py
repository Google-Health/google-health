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
r"""Stage 4: Non-Collision Audit & Text-Only Anti-Leakage Gate.

Two-phase validation:
  Phase 4a: Checks each distractor for ground-truth collisions (is it
            also a valid answer?) against the muted clip. Colliding
            distractors are replaced; items that still lack enough
            collision-free distractors are rejected.
  Phase 4b: Tests if the item is solvable text-only. Items guessed correctly
            without video are rejected (anti-leakage gate).

Each item keeps the number of distractors requested in Stage 3
(--num_distractors there; 9, i.e. 10-option MCQs, by default).

Usage:
  python stage4_antileak.py \
    --input_file=/tmp/stage3_distractors.json \
    --clips_dir=/path/to/clips/ \
    --output_file=/tmp/stage4_validated.json
"""

import concurrent.futures
import json
import os
import random
import re
import threading
from typing import Any

from absl import app
from absl import flags
from absl import logging
import jinja2
import pydantic

import constraints
import content_utils
import language_model

_INPUT_FILE = flags.DEFINE_string(
    'input_file',
    '/tmp/stage3_distractors.json',
    'Path to Stage 3 distractors JSON output.',
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    '/tmp/stage4_validated.json',
    'Path to write validated items JSON.',
)
_CLIPS_DIR = flags.DEFINE_string(
    'clips_dir',
    None,
    'Directory containing the muted clips (<clip>_noaudio.mp4, as written by'
    ' stage0_discover.py) used for the Phase 4a collision audit.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    language_model.DEFAULT_MODEL,
    'Gemini model ID to use for both phases.',
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers',
    10,
    'Number of parallel worker threads.',
)

# Distractor count for Stage 3 items that do not record one: 9 distractors,
# i.e. 10-option MCQs (A–J), as in the paper.
_DEFAULT_NUM_DISTRACTORS = 9


def _option_labels(num_options: int) -> list[str]:
  """Returns the option labels 'A', 'B', ... for num_options options."""
  return [chr(ord('A') + i) for i in range(num_options)]


# ---------------------------------------------------------------------------
# Programmatic Temporal Overlap Check
# ---------------------------------------------------------------------------


def _parse_timestamp_range(text: str) -> tuple[float, float] | None:
  """Extracts (start_sec, end_sec) from 'Between MM:SS and MM:SS'."""
  matches = re.findall(r'(\d+):(\d+(?:\.\d+)?)', text)
  if len(matches) >= 2:
    m1, s1 = float(matches[0][0]), float(matches[0][1])
    m2, s2 = float(matches[1][0]), float(matches[1][1])
    return (m1 * 60 + s1, m2 * 60 + s2)
  return None


def _is_temporal_overlap(
    gt_text: str,
    distractor_text: str,
    min_gap_sec: float = 1.0,
    max_clip_sec: float = 0.0,
) -> bool:
  """Returns True if distractor timestamp overlaps GT range or exceeds clip."""
  d_range = _parse_timestamp_range(distractor_text)
  if not d_range:
    return False
  d_start, d_end = d_range
  if max_clip_sec > 0 and d_end > max_clip_sec:
    return True
  gt_range = _parse_timestamp_range(gt_text)
  if not gt_range:
    return False
  gt_start, gt_end = gt_range
  if max(gt_start, d_start) <= min(gt_end, d_end) + min_gap_sec:
    return True
  return False


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class DistractorCollisionAudit(constraints.BaseModel):
  """Audit result for a single distractor."""

  distractor_text: str = pydantic.Field(
      description='The distractor text being audited.'
  )
  is_valid_answer_to_stem: bool = pydantic.Field(
      description=(
          'True ONLY if this distractor is ALSO a valid answer to the question'
          ' stem (ground-truth collision). False if it is visually false for'
          ' this target.'
      )
  )
  reasoning: str = pydantic.Field(
      description='Brief reasoning justifying the collision verdict.'
  )


@pydantic.dataclasses.dataclass
class ItemCollisionAuditBatch(constraints.BaseModel):
  """Audit results for the candidate distractors of an item."""

  audits: list[DistractorCollisionAudit] = pydantic.Field(
      description='Audit for each candidate distractor.'
  )


@pydantic.dataclasses.dataclass
class TextOnlyAnswer(constraints.BaseModel):
  """Model's text-only answer attempt."""

  reasoning: str = pydantic.Field(
      description='Reasoning based ONLY on text and medical knowledge.'
  )
  selected_option: str = pydantic.Field(
      description='Selected option letter (e.g. A).'
  )
  confidence: str = pydantic.Field(
      description='Confidence: HIGH, MEDIUM, or LOW.'
  )


@pydantic.dataclasses.dataclass
class ReplacementDistractorsResult(constraints.BaseModel):
  """Replacement distractors generated for rejected collisions."""

  distractors: list[str] = pydantic.Field(
      description='List of replacement distractor texts.'
  )


# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------


def _load_prompt_template(template_name: str) -> jinja2.Template:
  """Loads a Jinja2 prompt template by name."""
  template_path = os.path.join(
      os.path.dirname(__file__), 'templates', f'{template_name}.jinja2'
  )
  with open(template_path, 'r') as f:
    template_str = f.read()
  return jinja2.Environment(
      undefined=jinja2.StrictUndefined,
      trim_blocks=True,
      lstrip_blocks=True,
      autoescape=False,
      keep_trailing_newline=True,
  ).from_string(template_str)


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------


def _run_collision_audit(
    question: str,
    gt_answer: str,
    distractors: list[str],
    clip_path: str,
    clip_dur_sec: float,
    idx: int,
    lm: language_model.LanguageModel,
    num_distractors: int = _DEFAULT_NUM_DISTRACTORS,
) -> list[str] | None:
  """Runs Phase 4a collision audit with up to 3 retries.

  Args:
    question: The question stem text.
    gt_answer: The ground-truth answer text.
    distractors: List of candidate distractor texts to audit.
    clip_path: Path to the video clip file.
    clip_dur_sec: Duration of the clip in seconds.
    idx: Item index for logging.
    lm: Language model instance.
    num_distractors: Number of collision-free distractors the item needs.

  Returns:
    List of collision-free distractor strings, or None if fewer than
    num_distractors clean distractors remain after retries.
  """
  candidate_pool = list(distractors)
  collision_free: list[str] = []
  # Distractors that got a verdict, clean or not. Retries only audit new
  # replacement distractors: auditing a rejected one again could let it in.
  audited: set[str] = set()
  max_retries = 3

  for retry_idx in range(max_retries + 1):
    to_audit = [d for d in candidate_pool if d.strip() not in audited]
    if not to_audit:
      break

    distractors_formatted = '\n'.join(
        f'{i+1}. {d}' for i, d in enumerate(to_audit)
    )
    audit_prompt = _load_prompt_template(
        'stage4_collision_audit_prompt'
    ).render(
        question=question,
        correct_answer=gt_answer,
        distractors_list=distractors_formatted,
    )
    audit_prompt += constraints.get_constraint_suffix(
        cls=ItemCollisionAuditBatch
    )

    try:
      with content_utils.media_part(lm.client, clip_path) as clip_part:
        audit_res, _ = lm.generate_thought_and_object(
            [clip_part, audit_prompt], cls=ItemCollisionAuditBatch
        )
      for audit in audit_res.audits:
        audited.add(audit.distractor_text.strip())
        if audit.is_valid_answer_to_stem:
          logging.info('  ⚠️ Collision: "%s"', audit.distractor_text[:40])
        elif _is_temporal_overlap(
            gt_answer, audit.distractor_text, max_clip_sec=clip_dur_sec
        ):
          logging.info(
              '  ⚠️ Temporal GT Overlap / Out of Bounds: "%s"',
              audit.distractor_text[:40],
          )
        else:
          clean_d = audit.distractor_text.strip()
          if clean_d != gt_answer.strip() and clean_d not in [
              c.strip() for c in collision_free
          ]:
            collision_free.append(clean_d)
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Error in 4a for item %d: %s', idx, e)

    if len(collision_free) >= num_distractors:
      break

    if retry_idx < max_retries:
      _generate_replacement_distractors(
          question,
          gt_answer,
          collision_free,
          candidate_pool,
          num_distractors - len(collision_free),
          lm,
      )

  if len(collision_free) < num_distractors:
    logging.info(
        '  Item %d failed 4a after 3 retries (only %d of %d clean'
        ' distractors).',
        idx,
        len(collision_free),
        num_distractors,
    )
    return None

  return collision_free


def _generate_replacement_distractors(
    question: str,
    gt_answer: str,
    collision_free: list[str],
    candidate_pool: list[str],
    needed: int,
    lm: language_model.LanguageModel,
) -> None:
  """Generates replacement distractors and adds them to candidate_pool."""
  existing_str = '\n'.join(f'- {d}' for d in [gt_answer] + collision_free)
  replace_prompt = _load_prompt_template(
      'stage4_replace_distractors_prompt'
  ).render(
      question=question,
      correct_answer=gt_answer,
      existing_distractors=existing_str,
      num_needed=needed,
  )
  replace_prompt += constraints.get_constraint_suffix(
      cls=ReplacementDistractorsResult
  )
  try:
    rep_res, _ = lm.generate_thought_and_object(
        replace_prompt, cls=ReplacementDistractorsResult
    )
    for rep_d in rep_res.distractors:
      rep_clean = rep_d.strip()
      if (
          rep_clean not in [c.strip() for c in candidate_pool]
          and rep_clean not in [c.strip() for c in collision_free]
          and rep_clean != gt_answer.strip()
      ):
        candidate_pool.append(rep_clean)
  except Exception as e:  # pylint: disable=broad-except
    logging.error('Error generating replacements: %s', e)


def _run_antileak_gate(
    question: str,
    gt_answer: str,
    collision_free: list[str],
    idx: int,
    lm: language_model.LanguageModel,
    item: dict[str, Any],
    num_distractors: int = _DEFAULT_NUM_DISTRACTORS,
) -> dict[str, Any] | None:
  """Runs Phase 4b text-only anti-leakage gate.

  Args:
    question: The question stem text.
    gt_answer: The ground-truth answer text.
    collision_free: List of collision-free distractor texts from Phase 4a.
    idx: Item index for logging and random seeding.
    lm: Language model instance.
    item: Original item dict to copy metadata from.
    num_distractors: Number of distractors in the final MCQ.

  Returns:
    Validated item dict with options and audit status, or None if the item
    leaked (was correctly answered text-only) or could not be checked.
  """
  # Phase 4b: Text-only anti-leakage gate
  seen_texts: set[str] = set()
  deduped_distractors: list[str] = []
  for d in collision_free:
    d_clean = d.strip()
    if d_clean != gt_answer.strip() and d_clean not in seen_texts:
      seen_texts.add(d_clean)
      deduped_distractors.append(d)

  options_list = [gt_answer] + deduped_distractors[:num_distractors]
  random.seed(idx)
  random.shuffle(options_list)
  option_labels = _option_labels(len(options_list))
  correct_key = option_labels[options_list.index(gt_answer)]

  options_formatted = '\n'.join(
      f'{option_labels[i]}. {opt}' for i, opt in enumerate(options_list)
  )
  leak_prompt = _load_prompt_template('stage4_text_only_prompt').render(
      question=question,
      options_text=options_formatted,
      num_options=len(options_list),
      last_option_label=option_labels[-1],
  )
  leak_prompt += constraints.get_constraint_suffix(cls=TextOnlyAnswer)

  try:
    answer, _ = lm.generate_thought_and_object(leak_prompt, cls=TextOnlyAnswer)
  except Exception as e:  # pylint: disable=broad-except
    # Fail closed: an item that could not be checked must not pass the gate.
    logging.error('Error in 4b for item %d; rejecting it: %s', idx, e)
    return None
  if answer.selected_option.strip().upper() == correct_key:
    logging.info('  ❌ Item %d leaked text-only (key %s).', idx, correct_key)
    return None
  logging.info('  ✓ Item %d passed Stage 4.', idx)

  item_copy = dict(item)
  item_copy['collision_audit_status'] = 'PASS'
  item_copy['gate_status'] = 'PASS'
  item_copy['correct_key'] = correct_key
  item_copy['options'] = [
      {'key': option_labels[i], 'text': opt, 'is_correct': opt == gt_answer}
      for i, opt in enumerate(options_list)
  ]
  item_copy['num_options'] = len(options_list)
  return item_copy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _process_item(
    idx: int,
    item: dict[str, Any],
    lm: language_model.LanguageModel,
    clips_dir: str,
) -> dict[str, Any] | None:
  """Processes a single item through Phase 4a and Phase 4b."""
  question = item['question']
  gt_answer = item['correct_answer']
  distractors = item['distractors']
  clip_file = item.get('clip_file', '')
  num_distractors = item.get('num_distractors') or _DEFAULT_NUM_DISTRACTORS

  base_name = clip_file.replace('.mp4', '').replace('_noaudio', '')
  clip_path = os.path.join(clips_dir, f'{base_name}_noaudio.mp4')
  if not os.path.exists(clip_path):
    logging.error('Item %d: muted clip not found: %s', idx, clip_path)
    return None
  clip_dur_sec = item.get('clip_duration_sec', 0.0)

  collision_free = _run_collision_audit(
      question,
      gt_answer,
      distractors,
      clip_path,
      clip_dur_sec,
      idx,
      lm,
      num_distractors=num_distractors,
  )
  if collision_free is None:
    return None

  return _run_antileak_gate(
      question,
      gt_answer,
      collision_free,
      idx,
      lm,
      item,
      num_distractors=num_distractors,
  )


def _load_stage3_data(input_file: str) -> list[dict[str, Any]]:
  """Loads JSON and extracts 'items'."""
  logging.info('Loading Stage 3 items from %s...', input_file)
  with open(input_file, 'r') as f:
    input_items = json.load(f).get('items', [])
  logging.info('Found %d items to validate.', len(input_items))
  return input_items


def _save_checkpoint(
    output_file: str,
    input_count: int,
    items: list[dict[str, Any]],
) -> None:
  """Saves the summary dict to output JSON."""
  summary = {
      'stage': 'stage4_validated',
      'input_items': input_count,
      'passed_items': len(items),
      'validated_items': items,
  }
  with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)


def _run_parallel_validation(
    input_items: list[dict[str, Any]],
    lm: language_model.LanguageModel,
    clips_dir: str,
    num_workers: int,
    output_file: str,
) -> list[dict[str, Any]]:
  """Runs parallel validation using ThreadPoolExecutor."""
  validated_items: list[dict[str, Any]] = []
  save_lock = threading.Lock()

  def _task(args: tuple[int, dict[str, Any]]) -> None:
    idx, item = args
    res = _process_item(idx, item, lm, clips_dir)
    if res is not None:
      with save_lock:
        validated_items.append(res)
        # Write incremental checkpoint to output_file
        _save_checkpoint(output_file, len(input_items), validated_items)

  effective_workers = min(num_workers, len(input_items)) if input_items else 1
  logging.info('Running Stage 4 with %d worker threads...', effective_workers)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=effective_workers
  ) as executor:
    list(executor.map(_task, enumerate(input_items)))

  return validated_items


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  assert _CLIPS_DIR.value is not None

  input_items = _load_stage3_data(_INPUT_FILE.value)

  lm = language_model.LanguageModel(model_name=_MODEL_NAME.value)

  validated_items = _run_parallel_validation(
      input_items=input_items,
      lm=lm,
      clips_dir=_CLIPS_DIR.value,
      num_workers=_NUM_WORKERS.value,
      output_file=_OUTPUT_FILE.value,
  )

  logging.info(
      'Done! Saved %d validated items to %s',
      len(validated_items),
      _OUTPUT_FILE.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required(['clips_dir'])
  app.run(main)
