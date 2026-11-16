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
r"""Stage 3: Domain Co-Hyponym Distractor Generation.

Generates hard-negative distractors for each EIM question item. The number
of distractors is controlled by --num_distractors; the default of 9
distractors (i.e. 10-option MCQs, A–J) is the setting used in the paper.
Includes facet-specific rules for SPATIAL, ABSENCE, and TEMPORAL question
types.

Usage:
  python stage3_distractors.py \
    --questions_file=/tmp/stage2_questions.json \
    --output_file=/tmp/stage3_distractors.json \
    --num_distractors=9
"""

import concurrent.futures
import json
import os
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

_QUESTIONS_FILE = flags.DEFINE_string(
    'questions_file',
    '/tmp/stage2_questions.json',
    'Path to Stage 2 questions JSON output.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    language_model.DEFAULT_MODEL,
    'Gemini model ID to use.',
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    '/tmp/stage3_distractors.json',
    'Path to write the generated multiple-choice items JSON.',
)
_MAX_CLIPS = flags.DEFINE_integer(
    'max_clips',
    0,
    'Max clips to process (0 = all).',
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers',
    10,
    'Number of parallel worker threads.',
)
_CLIPS_DIR = flags.DEFINE_string(
    'clips_dir',
    '',  # directory containing local clips
    'Directory containing video clips. Used for ABSENCE distractor'
    ' generation to ground distractors in visible findings.',
)
_NUM_DISTRACTORS = flags.DEFINE_integer(
    'num_distractors',
    9,
    'Number of hard-negative distractors to generate per question item. The'
    ' resulting MCQ has num_distractors + 1 options, labeled A, B, C, ... The'
    ' default of 9 distractors (i.e. 10-option MCQs, A–J) is the setting used'
    ' in the paper. Stage 4 keeps exactly this many distractors per item.',
    lower_bound=1,
    upper_bound=25,
)


_DISTRACTOR_TEMPLATE_NAME = 'stage3_distractor_prompt'


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class CoHyponymDistractorCandidate(constraints.BaseModel):
  """A single domain co-hyponym distractor."""

  distractor_text: str = pydantic.Field(
      description=(
          'Distractor option text. Must be from the EXACT SAME clinical'
          ' domain and match word count (+/- 10%) and syntactic template of the'
          ' correct answer.'
      )
  )


@pydantic.dataclasses.dataclass
class CoHyponymGenerationResult(constraints.BaseModel):
  """Generated domain co-hyponym distractors for a question item."""

  domain_category: str = pydantic.Field(
      description=(
          'Anatomical/clinical domain (e.g. HAND_WRIST, THORACIC, ABDOMINAL).'
      )
  )
  distractors: list[CoHyponymDistractorCandidate] = pydantic.Field(
      description=(
          'List of domain co-hyponym distractors, containing exactly the'
          ' number of distractors requested in the prompt.'
      )
  )


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _load_prompt_template(
    template_name: str = _DISTRACTOR_TEMPLATE_NAME,
) -> jinja2.Template:
  """Loads a Jinja2 prompt template by name."""
  template_path = os.path.join(
      os.path.dirname(__file__), 'templates', f'{template_name}.jinja2'
  )
  with open(template_path, 'r') as f:
    template_str = f.read()
  env = jinja2.Environment(
      undefined=jinja2.StrictUndefined,
      trim_blocks=True,
      lstrip_blocks=True,
      autoescape=False,
      keep_trailing_newline=True,
  )
  return env.from_string(template_str)


# ---------------------------------------------------------------------------
# Question processing
# ---------------------------------------------------------------------------


def _process_question(
    q_tuple: tuple[dict[str, Any], dict[str, Any]],
    lm: language_model.LanguageModel,
    num_distractors: int,
) -> dict[str, Any] | None:
  """Generates distractors for a single question item."""
  clip_item, q = q_tuple
  question_stem = q.get('question')
  correct_ans = q.get('correct_answer')
  if not question_stem or not correct_ans:
    return None

  clip_file = clip_item.get('clip')
  condition = clip_item.get('condition', '')
  case_id = clip_item.get('case_id', '')
  clip_dur_str = clip_item.get('clip_duration_str', '00:30')
  clip_dur_sec = clip_item.get('clip_duration_sec', 30.0)

  template = _load_prompt_template()
  prompt = template.render(
      question=question_stem,
      correct_answer=correct_ans,
      eim_facet=q.get('eim_facet', 'ACTION'),
      target_anchor=q.get('target_anchor', ''),
      evidence_basis=q.get('evidence_basis', ''),
      clip_duration=clip_dur_str,
      num_distractors=num_distractors,
      num_options=num_distractors + 1,
  )
  prompt += constraints.get_constraint_suffix(cls=CoHyponymGenerationResult)

  try:
    is_absence = q.get('eim_facet', '').upper() == 'ABSENCE'

    if is_absence and _CLIPS_DIR.value:
      # Video-grounded generation for ABSENCE: the model watches the
      # clip at high FPS so it can identify findings that ARE present
      # and use them as distractors.
      clip_path = os.path.join(str(_CLIPS_DIR.value), str(clip_file))
      with content_utils.media_part(lm.client, clip_path) as clip_part:
        res, _ = lm.generate_thought_and_object(
            [clip_part, prompt], cls=CoHyponymGenerationResult
        )
    else:
      # Text-only generation for non-ABSENCE facets: ACTION, SPATIAL,
      # TEMPORAL, and APPEARANCE questions are grounded in the Stage 1
      # evidence text, so the prompt already encodes the visual
      # observations. Only ABSENCE needs the raw video because the
      # model must independently verify which findings are visually
      # present to avoid generating ambiguous distractors.
      res, _ = lm.generate_thought_and_object(
          prompt, cls=CoHyponymGenerationResult
      )
    distractor_texts = [d.distractor_text for d in res.distractors]
    logging.info(
        '  Generated %d distractors for [%s] %s',
        len(res.distractors),
        q.get('eim_facet'),
        question_stem[:40],
    )
    return {
        'clip_file': clip_file,
        'case_id': case_id,
        'condition': condition,
        'symptom': clip_item.get('symptom', ''),
        'clip_duration_sec': clip_dur_sec,
        'clip_duration_str': clip_dur_str,
        'eim_facet': q.get('eim_facet', 'ACTION').upper(),
        'question_type': q.get('question_type', 'action'),
        'target_anchor': q.get('target_anchor', ''),
        'question': question_stem,
        'correct_answer': correct_ans,
        'evidence_basis': q.get('evidence_basis', ''),
        'distractors': distractor_texts,
        'domain_category': res.domain_category,
        'num_options': 1 + len(res.distractors),
        # The requested count; Stage 4 keeps exactly this many distractors.
        'num_distractors': num_distractors,
    }
  except Exception as e:  # pylint: disable=broad-except
    logging.error('Error generating distractors for %s: %s', question_stem, e)
    return None


# ---------------------------------------------------------------------------
# Main helpers
# ---------------------------------------------------------------------------


def _load_stage2_data(
    questions_file: str, max_clips: int = 0
) -> list[dict[str, Any]]:
  """Loads Stage 2 questions JSON and limits to max_clips if specified."""
  logging.info('Loading Stage 2 questions from %s...', questions_file)
  with open(questions_file, 'r') as f:
    stage2_data = json.load(f)
  clip_results = stage2_data.get('results', [])
  if max_clips > 0:
    clip_results = clip_results[:max_clips]
  return clip_results


def _build_work_items(
    clip_results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
  """Flattens clip questions into work item tuples."""
  work_items = []
  for clip_item in clip_results:
    for q in clip_item.get('questions', []):
      work_items.append((clip_item, q))
  return work_items


def _save_summary(
    output_file: str,
    model_name: str,
    items: list[dict[str, Any]],
) -> None:
  """Saves current generated distractors to output JSON."""
  summary = {
      'model': model_name,
      'stage': 'stage3_cohyponym_distractors',
      'total_items': len(items),
      'items': items,
  }
  with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)


def _run_parallel_generation(
    work_items: list[tuple[dict[str, Any], dict[str, Any]]],
    lm: language_model.LanguageModel,
    num_workers: int,
    output_file: str,
    model_name: str,
    num_distractors: int,
) -> list[dict[str, Any]]:
  """Runs distractor generation across worker threads and saves results."""
  all_items: list[dict[str, Any]] = []
  save_lock = threading.Lock()

  def _task(q_tuple: tuple[dict[str, Any], dict[str, Any]]) -> None:
    res = _process_question(q_tuple, lm, num_distractors)
    if res is not None:
      with save_lock:
        all_items.append(res)
        _save_summary(output_file, model_name, all_items)

  actual_workers = min(num_workers, len(work_items)) if work_items else 1
  logging.info(
      'Processing %d question items with %d worker threads...',
      len(work_items),
      actual_workers,
  )

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=actual_workers
  ) as executor:
    list(executor.map(_task, work_items))

  return all_items


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  clip_results = _load_stage2_data(
      _QUESTIONS_FILE.value, max_clips=_MAX_CLIPS.value
  )
  work_items = _build_work_items(clip_results)

  lm = language_model.LanguageModel(model_name=_MODEL_NAME.value)

  all_items = _run_parallel_generation(
      work_items=work_items,
      lm=lm,
      num_workers=_NUM_WORKERS.value,
      output_file=_OUTPUT_FILE.value,
      model_name=_MODEL_NAME.value,
      num_distractors=_NUM_DISTRACTORS.value,
  )

  logging.info(
      'Stage 3 distractors (%d items) written to %s',
      len(all_items),
      _OUTPUT_FILE.value,
  )


if __name__ == '__main__':
  app.run(main)
