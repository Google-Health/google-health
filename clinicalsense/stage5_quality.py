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
r"""Stage 5: Quality Gates & Final Benchmark Formatting.

Formats validated Stage 4 items into final ClinicalSense benchmark JSON:
  1. Preserves the audited, deduplicated A–J options and keys from Stage 4.
  2. Filters out items from clips shorter than --min_clip_duration seconds.
  3. Assigns standard benchmark item IDs (e.g. CS-ACTION-001).
  4. Aggregates summary statistics across facets and conditions.

Usage:
  python stage5_quality.py \
    --input_file=/tmp/stage4_validated.json \
    --output_file=/tmp/clinicalsense_benchmark.json
"""

import collections
import json
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging

_INPUT_FILE = flags.DEFINE_string(
    'input_file',
    '/tmp/stage4_validated.json',
    'Path to Stage 4 validated output JSON.',
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    '/tmp/clinicalsense_benchmark.json',
    'Path to write the final benchmark JSON.',
)
_MIN_CLIP_DURATION = flags.DEFINE_float(
    'min_clip_duration',
    3.0,
    'Minimum clip duration in seconds. Items from shorter clips are excluded.',
)


def _format_benchmark_item(
    item: dict[str, Any], item_idx: int
) -> dict[str, Any] | None:
  """Preserve validated Stage 4 options and assign benchmark item ID."""
  correct_answer = item.get('correct_answer')
  if not correct_answer:
    return None

  eim_facet = item.get('eim_facet', 'ACTION').upper()
  options = item.get('options', [])
  correct_key = item.get('correct_key', '')

  # Fallback formatting if options were not pre-assembled
  if not options:
    distractors = item.get('distractors', [])
    seen = set()
    unique_options = []
    for opt in [correct_answer] + distractors:
      if opt is None:
        continue
      normalized = opt.strip().lower()
      if normalized not in seen:
        seen.add(normalized)
        unique_options.append(opt.strip())
    options = []
    for j, opt in enumerate(unique_options):
      key = chr(65 + j)
      is_corr = opt.strip().lower() == correct_answer.strip().lower()
      if is_corr:
        correct_key = key
      options.append({'key': key, 'text': opt, 'is_correct': is_corr})

  return {
      'item_id': f'CS-{eim_facet}-{item_idx:03d}',
      'clip_file': item['clip_file'],
      'case_id': item['case_id'],
      'condition': item['condition'],
      'symptom': item.get('symptom', ''),
      'clip_duration_sec': item.get('clip_duration_sec', 0.0),
      'eim_facet': eim_facet,
      'question_type': item.get('question_type', 'action'),
      'target_anchor': item.get('target_anchor', ''),
      'question': item['question'],
      'options': options,
      'correct_key': correct_key,
      'correct_answer': correct_answer,
      'evidence_basis': item.get('evidence_basis', ''),
      'num_options': len(options),
      'domain_category': item.get('domain_category', ''),
  }


def _filter_by_clip_duration(
    items: list[dict[str, Any]], min_dur: float
) -> list[dict[str, Any]]:
  """Filters items from clips shorter than the minimum duration.

  Args:
    items: List of validated item dicts from Stage 4.
    min_dur: Minimum clip duration in seconds.

  Returns:
    Filtered list containing only items from clips >= min_dur seconds.
  """
  if min_dur <= 0:
    return items

  before = len(items)
  filtered = [it for it in items if it.get('clip_duration_sec', 0.0) >= min_dur]
  dropped = before - len(filtered)
  if dropped:
    logging.info(
        'Filtered %d items from clips < %.0fs (%d remaining).',
        dropped,
        min_dur,
        len(filtered),
    )
  return filtered


def _build_summary(
    final_items: list[dict[str, Any]], min_dur: float
) -> dict[str, Any]:
  """Builds the benchmark summary dict with statistics and items.

  Args:
    final_items: List of formatted benchmark item dicts.
    min_dur: Minimum clip duration used for filtering.

  Returns:
    Summary dict containing benchmark metadata, facet/condition counts,
    and the final items list.
  """
  facet_counts = collections.Counter(it['eim_facet'] for it in final_items)
  condition_counts = collections.Counter(it['condition'] for it in final_items)

  return {
      'benchmark_name': 'ClinicalSense',
      'total_items': len(final_items),
      'min_clip_duration_sec': min_dur,
      'by_eim_facet': dict(facet_counts),
      'by_condition': dict(condition_counts),
      'items': final_items,
  }


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Loading Stage 4 output from %s...', _INPUT_FILE.value)
  with open(_INPUT_FILE.value, 'r') as f:
    validated_items = json.load(f).get('validated_items', [])
  logging.info('Found %d validated items.', len(validated_items))

  # Filter by minimum clip duration
  min_dur = _MIN_CLIP_DURATION.value
  validated_items = _filter_by_clip_duration(validated_items, min_dur)

  final_items = []
  for idx, item in enumerate(validated_items):
    formatted = _format_benchmark_item(item, idx + 1)
    if formatted is not None:
      final_items.append(formatted)

  summary = _build_summary(final_items, min_dur)

  with open(_OUTPUT_FILE.value, 'w') as f:
    json.dump(summary, f, indent=2)

  logging.info(
      'Benchmark: %d items written to %s', len(final_items), _OUTPUT_FILE.value
  )
  facet_counts = summary['by_eim_facet']
  logging.info(
      'ClinicalSense benchmark: %d items (%s)',
      len(final_items),
      ', '.join(f'{k}={v}' for k, v in sorted(facet_counts.items())),
  )


if __name__ == '__main__':
  app.run(main)
