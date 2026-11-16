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
r"""Stage 2: EIM Question Generation from Visual Evidence.

Takes Stage 1 visual evidence (normally the output of stage1b_patch.py) and
generates one clinically-grounded MCQ per EIM perception facet (ACTION,
SPATIAL, TEMPORAL, APPEARANCE, ABSENCE).

Usage:
  python stage2_question_gen.py \
    --evidence_file=/tmp/stage1b_patched.json \
    --output_file=/tmp/stage2_questions.json
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
import language_model

_EVIDENCE_FILE = flags.DEFINE_string(
    'evidence_file',
    '/tmp/stage1_visual_evidence.json',
    'Path to Stage 1 visual evidence JSON output.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    language_model.DEFAULT_MODEL,
    'Gemini model ID to use.',
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    '/tmp/stage2_questions.json',
    'Path to write the generated questions JSON.',
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers',
    10,
    'Number of parallel worker threads.',
)
_MIN_EVIDENCE_ITEMS = flags.DEFINE_integer(
    'min_evidence_items',
    1,
    'Minimum required successful evidence items to proceed (logs warning if'
    ' below).',
)


_QA_GEN_TEMPLATE_NAME = 'stage2_question_gen_prompt'

# ---------------------------------------------------------------------------
# EIM Facet Definitions — question generation instructions
# ---------------------------------------------------------------------------

_FACET_KEYS = ('action', 'spatial', 'temporal', 'appearance', 'absence')

EIM_FACETS = {
    'action': {
        'name': 'Action (Intentionality)',
        'clinical_significance': (
            'Guarding, bracing, splinting, rubbing, and compensatory movements'
            ' indicate pain localization and severity (Bates Physical Exam).'
        ),
        'question_instruction': (
            'Generate a CONCISE, NATURAL question about a physical action or'
            ' maneuver.\n'
            ' Exemplar stems (Bates & OSCE Physical Exam Inspection):\n'
            '   - "Which physical guarding or splinting behavior does the'
            ' patient display during [movement/cough]?"\n'
            '   - "What protective maneuver does the patient use to support'
            ' their upper body when changing position?"\n'
            '   - "How does the patient position their arm or hand to protect'
            ' the painful area during [maneuver]?"\n'
            ' Keep stem under 20 words. Anchor to a body part or action'
            ' context.\n'
            ' USE standard clinical inspection terms: place hand on, brace,'
            ' guard, rub, point to, lean forward, support.\n'
            ' BAN synthetic LLM filler: "protective physical maneuver",'
            ' "deliberate forward bending motion", "upper extremity'
            ' positioning".'
        ),
    },
    'spatial': {
        'name': 'Spatial (Space)',
        'clinical_significance': (
            'Anatomical localization of guarding, pointing, or bracing'
            ' informs differential diagnosis (Bates Physical Exam).'
        ),
        'question_instruction': (
            'Generate a CONCISE, NATURAL question about anatomical location,'
            ' direction, or laterality.\n Exemplar stems (Bates & OSCE'
            ' Anatomical Localization):\n   - "In which anatomical region'
            ' (e.g., epigastrium, right upper quadrant) does the patient place'
            ' their hand?"\n   - "On which side of the body (laterality) is the'
            ' physical finding or guarding localized?"\n   - "Where on the'
            ' torso or limb is the visual finding centered?"\n Keep stem under'
            ' 20 words. Anchor to a clear spatial target.\n USE standard'
            ' clinical inspection terms: left lower quadrant, mid-chest, right'
            ' shoulder, sternal border, epigastrium, flank.\n BAN synthetic LLM'
            ' filler: "anatomical spatial locus", "laterality index", "spatial'
            ' orientation quadrant".'
        ),
    },
    'temporal': {
        'name': 'Temporal (Time)',
        'clinical_significance': (
            'Timing of symptom onset/provocation, sequence of movements, and'
            ' sign duration distinguish acute pain behaviors.'
        ),
        'question_instruction': (
            'Generate a CONCISE, NATURAL question about timing or event'
            ' sequence.\n'
            ' If timestamps span ≥10s, generate a TIMESTAMP question (OSCE'
            ' Provocation & Timing):\n'
            '   - Stem: "At what timestamp in the clip (e.g., Between 0:12 and'
            ' 0:14) does [event] occur?"\n'
            '   - Answer MUST be exact timestamp range (e.g., "Between 0:12 and'
            ' 0:14").\n'
            ' If clip is SHORT (<10s), generate a SEQUENCE question (OSCE'
            ' Maneuver Sequence):\n'
            '   - Stem: "What physical response immediately follows the'
            ' instruction to [maneuver]?"\n'
            ' Keep stem under 20 words.\n'
            ' USE standard clinical inspection terms: At what point in the'
            ' clip..., Between MM:SS and MM:SS, immediately after, prior to.'
            ' BAN synthetic LLM filler: "temporal timestamp window of'
            ' occurrence", "chronological delta interval", "sequence'
            ' trajectory onset".'
        ),
    },
    'appearance': {
        'name': 'Appearance (Entity State)',
        'clinical_significance': (
            'Facial expression, eye state, posture, and respiratory signs'
            ' indicate distress level (Bates End-of-Bed General Survey).'
        ),
        'question_instruction': (
            'Generate a CONCISE, NATURAL question about a visual attribute.\n'
            ' Exemplar stems (Bates End-of-Bed General Survey Inspection):\n'
            '   - "What is the patient\'s facial expression or sign of'
            ' distress (e.g., grimacing, wincing) during the encounter?"\n'
            '   - "What is the state of the patient\'s eyes (open, closed,'
            ' squinting) during the movement?"\n'
            '   - "Which overall body posture (e.g., tripod position, hunched)'
            ' does the patient display?"\n'
            ' Keep stem under 20 words. PRIORITIZE: facial distress,'
            ' eye state, overall body posture, skin/respiratory signs.\n'
            ' Do NOT ask about fine-grained hand/finger posture (e.g. "how'
            ' are fingers positioned") — this is resolution testing.\n'
            ' USE standard clinical inspection terms: facial expression,'
            ' grimace, eyes closed/open, hunched posture, accessory muscle'
            ' use.\n'
            ' BAN synthetic LLM filler: "visual attribute matrix", "entity'
            ' state manifestation", "facial musculature contraction'
            ' phenomenon".'
        ),
    },
    'absence': {
        'name': 'Absence (Causation / Expectation Violation)',
        'clinical_significance': (
            'Documenting absent expected physical signs narrows the'
            ' differential diagnosis (DeGowin Pertinent Negatives).'
        ),
        'question_instruction': (
            'Generate a CONCISE, NATURAL question testing observation of'
            ' ABSENT signs.\n'
            ' Exemplar stems (OSCE Pertinent Negatives on Inspection):\n'
            '   - "Which of the following physical findings is noticeably'
            ' absent during the general inspection?"\n'
            '   - "Which expected physical indicator of distress (e.g.,'
            ' accessory muscle use, facial grimacing) is NOT observed?"\n'
            '   - "Which expected protective maneuver or sign does the'
            ' patient FAIL to display?"\n'
            ' Keep stem under 20 words. Ground truth answer MUST be a'
            ' clinically plausible but visually ABSENT finding.\n'
            ' USE standard clinical inspection terms: Which of the following'
            ' physical findings is NOT observed...?, is noticeably absent.'
            ' BAN synthetic LLM filler: "expectation violation parameter",'
            ' "negative physical entity absence", "non-manifesting feature".'
        ),
    },
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class GeneratedQuestion(constraints.BaseModel):
  """A single generated question with correct answer."""

  eim_facet: str = pydantic.Field(
      description=(
          'Which EIM facet: ACTION, SPATIAL, TEMPORAL, APPEARANCE, or ABSENCE.'
      )
  )
  target_anchor: str = pydantic.Field(
      description=(
          'Physical target (e.g., "right hand position", "facial grimace").'
      )
  )
  question: str = pydantic.Field(
      description='Concise question stem (max 20 words). Must end with "?".'
  )
  correct_answer: str = pydantic.Field(
      description='Precise ground-truth answer based on visual evidence.'
  )
  evidence_basis: str = pydantic.Field(
      description='Visual evidence supporting this answer (1-2 sentences).'
  )


@pydantic.dataclasses.dataclass
class QuestionBatch(constraints.BaseModel):
  """A batch of generated questions across EIM facets for a single clip."""

  questions: list[GeneratedQuestion] = pydantic.Field(
      description='List of generated questions (one per EIM facet).'
  )


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _build_facet_instructions() -> str:
  """Builds numbered facet instruction text from EIM_FACETS."""
  return '\n'.join(
      f'{i}. [{k.upper()}] {EIM_FACETS[k]["name"]}:\n'
      f'   {EIM_FACETS[k]["question_instruction"]}'
      for i, k in enumerate(_FACET_KEYS, 1)
  )


def _load_prompt_template(
    template_name: str = _QA_GEN_TEMPLATE_NAME,
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
# Clip processing
# ---------------------------------------------------------------------------


def _process_clip(
    item: dict[str, Any],
    lm: language_model.LanguageModel,
    facet_instructions: str,
) -> dict[str, Any] | None:
  """Generates questions for a single clip item."""
  evidence = item.get('evidence')
  clip_name = item.get('clip')
  if not evidence or not clip_name:
    return None

  template = _load_prompt_template()
  prompt = template.render(
      primary_sign=evidence.get('primary_sign', ''),
      action=evidence.get('action', ''),
      spatial=evidence.get('spatial', ''),
      temporal=evidence.get('temporal', ''),
      appearance=evidence.get('appearance', ''),
      absence=evidence.get('absence', ''),
      num_facets=len(_FACET_KEYS),
      facet_instructions=facet_instructions,
  )
  prompt += constraints.get_constraint_suffix(cls=QuestionBatch)

  try:
    batch, _ = lm.generate_thought_and_object(prompt, cls=QuestionBatch)
    generated_qs = [
        {
            'eim_facet': (
                q.eim_facet.upper()
                if getattr(q, 'eim_facet', None)
                else _FACET_KEYS[idx].upper()
            ),
            'question_type': (
                q.eim_facet.lower()
                if getattr(q, 'eim_facet', None)
                else _FACET_KEYS[idx]
            ),
            'question': q.question,
            'correct_answer': q.correct_answer,
            'target_anchor': q.target_anchor,
            'evidence_basis': q.evidence_basis,
        }
        for idx, q in enumerate(batch.questions)
    ]
    logging.info(
        '  Generated %d questions for %s', len(generated_qs), clip_name
    )
    return {
        'clip': clip_name,
        'case_id': item.get('case_id', ''),
        'condition': item.get('condition', ''),
        'symptom': item.get('symptom', ''),
        'clip_duration_sec': item.get('clip_duration_sec', 30.0),
        'clip_duration_str': item.get('clip_duration_str', '00:30'),
        'questions': generated_qs,
    }
  except Exception as e:  # pylint: disable=broad-except
    logging.error('Error generating questions for %s: %s', clip_name, e)
    return None


# ---------------------------------------------------------------------------
# Main helpers
# ---------------------------------------------------------------------------


def _load_stage1_evidence(
    evidence_file: str, min_items: int = 1
) -> list[dict[str, Any]]:
  """Loads Stage 1 visual evidence JSON and filters successful items."""
  logging.info('Loading Stage 1 evidence from %s...', evidence_file)
  with open(evidence_file, 'r') as f:
    stage1_data = json.load(f)

  successful_items = [
      r for r in stage1_data.get('results', []) if r.get('status') == 'success'
  ]
  logging.info('Found %d successful evidence items.', len(successful_items))

  if len(successful_items) < min_items:
    logging.warning(
        'Found %d successful evidence items, which is below the minimum'
        ' threshold of %d. Check Stage 1 output for failures.',
        len(successful_items),
        min_items,
    )
  return successful_items


def _save_results(
    output_file: str, model_name: str, results: list[dict[str, Any]]
) -> None:
  """Saves generated question results to output JSON."""
  output_data = {
      'model': model_name,
      'num_clips': len(results),
      'results': results,
  }
  with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)


def _run_parallel_generation(
    items: list[dict[str, Any]],
    lm: language_model.LanguageModel,
    facet_instructions: str,
    num_workers: int,
    output_file: str,
    model_name: str,
) -> list[dict[str, Any]]:
  """Runs question generation across worker threads with incremental saving."""
  all_results: list[dict[str, Any]] = []
  save_lock = threading.Lock()

  def _task(item: dict[str, Any]) -> None:
    res = _process_clip(item, lm, facet_instructions)
    if res is not None:
      with save_lock:
        all_results.append(res)
        _save_results(output_file, model_name, all_results)

  actual_workers = min(num_workers, len(items)) if items else 1
  logging.info('Running Stage 2 with %d worker threads...', actual_workers)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=actual_workers
  ) as executor:
    list(executor.map(_task, items))

  return all_results


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  successful_items = _load_stage1_evidence(
      _EVIDENCE_FILE.value, min_items=_MIN_EVIDENCE_ITEMS.value
  )
  if not successful_items:
    return

  lm = language_model.LanguageModel(model_name=_MODEL_NAME.value)

  facet_instructions = _build_facet_instructions()

  all_results = _run_parallel_generation(
      items=successful_items,
      lm=lm,
      facet_instructions=facet_instructions,
      num_workers=_NUM_WORKERS.value,
      output_file=_OUTPUT_FILE.value,
      model_name=_MODEL_NAME.value,
  )

  logging.info(
      'Stage 2 questions (%d clips) written to %s',
      len(all_results),
      _OUTPUT_FILE.value,
  )


if __name__ == '__main__':
  app.run(main)
