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
r"""Quick MCQ evaluator: run models on ClinicalSense benchmark items.

Sends video clip (no audio) + question + shuffled options to a model,
asks it to pick the correct answer, and scores accuracy.

Usage:
  python eval_mcq.py \
    --benchmark_file=/tmp/clinicalsense_benchmark.json \
    --clips_dir=/path/to/clips/ \
    --model_name=gemini-2.5-flash \
    --output_file=/tmp/eval_results.json
"""

import json
import os
import random
import subprocess
import tempfile
import traceback
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import pydantic

import constraints
import content_utils
import language_model

_BENCHMARK_FILE = flags.DEFINE_string(
    'benchmark_file',
    '/tmp/clinicalsense_benchmark.json',
    'Path to the benchmark JSON written by stage5_quality.py.',
)
_CLIPS_DIR = flags.DEFINE_string(
    'clips_dir',
    '',  # directory containing local clips
    'Directory containing video clips.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    language_model.DEFAULT_MODEL,
    'Gemini model ID to use.',
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    '/tmp/eval_results.json',
    'Path to write evaluation results.',
)
_INCLUDE_AUDIO = flags.DEFINE_bool(
    'include_audio',
    False,
    'If True, the model gets the clip with its audio track (video+audio eval).'
    ' If False, it gets the clip without its audio track and is told to'
    ' ignore audio (video-only eval).',
)
_TEXT_ONLY = flags.DEFINE_bool(
    'text_only',
    False,
    'If True, run text-only evaluation (no video or audio).',
)
_FPS = flags.DEFINE_float(
    'fps',
    5.0,
    'Frames per second for Gemini video decoding (default: 5.0 FPS).',
)
_AUDIO_ONLY = flags.DEFINE_bool(
    'audio_only',
    False,
    'If True, extracts audio from clip and sends audio + question (no video).',
)


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class MCQAnswer(constraints.BaseModel):
  """Model's answer to an MCQ question."""

  reasoning: str = pydantic.Field(
      description=(
          'Brief reasoning about what you observe in the video and why you'
          ' chose this answer (2-3 sentences).'
      )
  )
  selected_option: str = pydantic.Field(
      description=(
          'The letter of the selected option (e.g., "A", "B", "C", etc.).'
          ' Must be exactly one uppercase letter.'
      )
  )
  confidence: str = pydantic.Field(
      description='Your confidence level: HIGH, MEDIUM, or LOW.'
  )


# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

_EVAL_PROMPT = """You are taking a clinical visual perception exam.
Watch the video clip carefully and answer the following question.

IMPORTANT: Base your answer ONLY on what you can SEE in the video.
{audio_instruction}
Do NOT use general medical knowledge to guess — only answer based on
visual observation.

QUESTION: {question}

OPTIONS:
{options_text}

Select the single best answer.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_benchmark_items(benchmark_file: str) -> list[dict[str, Any]]:
  """Loads candidate or benchmark items from JSON file."""
  logging.info('Loading benchmark from %s...', benchmark_file)
  with open(benchmark_file, 'r') as f:
    bench_data = json.load(f)
  items = bench_data.get('items', [])
  logging.info('Loaded %d benchmark items.', len(items))
  return items


def _load_or_resume_checkpoint(
    output_file: str,
) -> tuple[list[dict[str, Any]], set[str]]:
  """Loads existing checkpoint if present, returning results and evaluated keys."""
  results: list[dict[str, Any]] = []
  evaluated_keys: set[str] = set()

  if os.path.exists(output_file):
    try:
      with open(output_file, 'r') as f:
        checkpoint = json.load(f)
      results = checkpoint.get('results', [])
      for r in results:
        key = f"{r.get('clip', '')}|{r.get('question_type', '')}"
        evaluated_keys.add(key)
      logging.info(
          'Loaded checkpoint with %d evaluated items.', len(evaluated_keys)
      )
    except (json.JSONDecodeError, KeyError) as e:
      logging.warning('Failed to load checkpoint: %s. Starting fresh.', e)
      results = []
      evaluated_keys = set()

  return results, evaluated_keys


def _save_checkpoint(
    path: str,
    results: list[dict[str, Any]],
    model_name: str,
    include_audio: bool,
    text_only: bool,
    audio_only: bool,
) -> None:
  """Saves evaluation checkpoint with current results."""
  correct = sum(1 for r in results if r.get('is_correct', False))
  total = len(results)
  checkpoint = {
      'model': model_name,
      'include_audio': include_audio,
      'text_only': text_only,
      'audio_only': audio_only,
      'total_items': total,
      'correct': correct,
      'accuracy': correct / total if total else 0,
      'results': results,
  }
  with open(path, 'w') as f:
    json.dump(checkpoint, f, indent=2)
  logging.info(
      'Checkpoint saved: %d/%d items (%.0f%%)',
      correct,
      total,
      100 * correct / total if total else 0,
  )


def _format_question_options(
    item: dict[str, Any],
) -> tuple[str, str, list[str]]:
  """Extracts formatted options text, correct key, and option list."""
  if 'options' in item:
    options_text = '\n'.join(
        f"{opt['key']}. {opt['text']}" for opt in item['options']
    )
    correct_key = item.get('correct_key', '')
    if not correct_key:
      for opt in item['options']:
        if opt.get('is_correct'):
          correct_key = opt['key']
          break
    all_options = [opt['text'] for opt in item['options']]
  else:
    all_options = [item['correct_answer']] + item.get('distractors', [])
    random.seed(hash(item['question']))
    random.shuffle(all_options)
    correct_idx = all_options.index(item['correct_answer'])
    correct_key = chr(65 + correct_idx)
    options_text = '\n'.join(
        f'{chr(65+j)}. {opt}' for j, opt in enumerate(all_options)
    )
  return options_text, correct_key, all_options


def _strip_audio(video_path: str, out_path: str) -> None:
  """Writes a copy of a video without its audio track."""
  subprocess.run(
      [
          'ffmpeg',
          '-y',
          '-i',
          video_path,
          '-map',
          '0:v:0',
          '-c:v',
          'copy',
          '-an',
          out_path,
      ],
      check=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
  )


def _evaluate_single_item(
    item: dict[str, Any],
    lm: language_model.LanguageModel,
    clips_dir: str,
    include_audio: bool,
    text_only: bool,
    audio_only: bool,
    fps: float | None = None,
) -> dict[str, Any] | None:
  """Evaluates model performance on a single benchmark MCQ item."""
  clip_path = os.path.join(clips_dir, item['clip_file'])

  if not text_only and not audio_only and not os.path.exists(clip_path):
    logging.warning('Clip not found: %s', clip_path)
    return None
  if audio_only and not os.path.exists(clip_path):
    logging.warning('Clip not found for audio extraction: %s', clip_path)
    return None

  options_text, correct_key, all_options = _format_question_options(item)
  audio_instruction = (
      ''
      if include_audio
      else 'IGNORE any audio — focus ONLY on visual information.'
  )

  prompt = _EVAL_PROMPT.format(
      question=item['question'],
      options_text=options_text,
      audio_instruction=audio_instruction,
  )
  prompt += constraints.get_constraint_suffix(cls=MCQAnswer)

  try:
    if text_only:
      text_prompt = (
          'Answer this clinical question using ONLY the question text'
          ' and your medical knowledge. NO video is provided.\n\n'
          f'QUESTION: {item["question"]}\n\nOPTIONS:\n{options_text}\n\n'
          'Select the single best answer.'
      )
      text_prompt += constraints.get_constraint_suffix(cls=MCQAnswer)
      answer, _ = lm.generate_thought_and_object(text_prompt, cls=MCQAnswer)
    elif audio_only:
      with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        local_audio = tmp.name
      try:
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-i',
                clip_path,
                '-vn',
                '-ac',
                '1',
                '-ar',
                '16000',
                '-acodec',
                'pcm_s16le',
                local_audio,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        audio_prompt = (
            'Listen to this audio clip from a clinical consultation. '
            'Answer the following question based on what you HEAR.\n\n'
            f'QUESTION: {item["question"]}\n\nOPTIONS:\n{options_text}\n\n'
            'Select the single best answer.'
        )
        audio_prompt += constraints.get_constraint_suffix(cls=MCQAnswer)
        with content_utils.media_part(lm.client, local_audio) as audio_part:
          answer, _ = lm.generate_thought_and_object(
              [audio_part, audio_prompt], cls=MCQAnswer
          )
      finally:
        os.unlink(local_audio)
    else:
      with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = clip_path
        if not include_audio:
          # Video-only: the model gets the clip without its audio track, rather
          # than only being told to ignore the audio.
          video_path = os.path.join(tmp_dir, os.path.basename(clip_path))
          _strip_audio(clip_path, video_path)
        with content_utils.media_part(
            lm.client, video_path, fps=fps
        ) as clip_part:
          answer, _ = lm.generate_thought_and_object(
              [clip_part, prompt], cls=MCQAnswer
          )

    is_correct = answer.selected_option.strip().upper() == correct_key
    return {
        'clip': item['clip_file'],
        'condition': item.get('condition', ''),
        'question_type': item.get('question_type', ''),
        'question': item['question'],
        'correct_key': correct_key,
        'correct_answer': item.get('correct_answer', ''),
        'model_selected': answer.selected_option.strip().upper(),
        'model_reasoning': answer.reasoning,
        'model_confidence': answer.confidence,
        'is_correct': is_correct,
        'num_options': len(all_options),
    }

  except Exception as e:  # pylint: disable=broad-except
    logging.error('Error evaluating %s: %s', item['clip_file'], e)
    logging.error(traceback.format_exc())
    return {
        'clip': item['clip_file'],
        'question_type': item.get('question_type', ''),
        'question': item['question'],
        'status': 'error',
        'error': str(e),
    }


def _run_evaluation_loop(
    items: list[dict[str, Any]],
    evaluated_keys: set[str],
    results: list[dict[str, Any]],
    lm: language_model.LanguageModel,
    clips_dir: str,
    output_file: str,
    model_name: str,
    include_audio: bool,
    text_only: bool,
    audio_only: bool,
    fps: float | None = None,
) -> list[dict[str, Any]]:
  """Runs evaluation loop across items with incremental checkpointing."""
  correct_count = sum(1 for r in results if r.get('is_correct', False))
  total_count = len(results)

  for i, item in enumerate(items):
    item_key = f"{item['clip_file']}|{item['question_type']}"
    if item_key in evaluated_keys:
      continue

    logging.info(
        '[%d/%d] %s [%s]',
        i + 1,
        len(items),
        item['clip_file'],
        item.get('question_type'),
    )
    res = _evaluate_single_item(
        item=item,
        lm=lm,
        clips_dir=clips_dir,
        include_audio=include_audio,
        text_only=text_only,
        audio_only=audio_only,
        fps=fps,
    )
    if res is not None:
      results.append(res)
      if res.get('is_correct'):
        correct_count += 1
      total_count += 1

      status = '✓' if res.get('is_correct') else '✗'
      logging.info(
          '  %s Model=%s (correct=%s) [%s] — %d/%d (%.0f%%)',
          status,
          res.get('model_selected', '?'),
          res.get('correct_key', '?'),
          res.get('model_confidence', '?'),
          correct_count,
          total_count,
          100 * correct_count / total_count if total_count else 0,
      )

    if len(results) % 5 == 0 or i == len(items) - 1:
      _save_checkpoint(
          output_file,
          results,
          model_name,
          include_audio,
          text_only,
          audio_only,
      )

  return results


def _compute_summary_stats(
    results: list[dict[str, Any]],
    model_name: str,
    include_audio: bool,
) -> dict[str, Any]:
  """Computes overall accuracy, random baseline, and breakdown metrics."""
  correct_count = sum(1 for r in results if r.get('is_correct', False))
  total_count = len(results)
  accuracy = correct_count / total_count if total_count else 0
  if total_count:
    random_baseline = (
        sum(1 / r['num_options'] for r in results if 'num_options' in r)
        / total_count
    )
  else:
    random_baseline = 0

  type_stats: dict[str, dict[str, int]] = {}
  for r in results:
    qt = r.get('question_type', 'unknown')
    type_stats.setdefault(qt, {'correct': 0, 'total': 0})
    type_stats[qt]['total'] += 1
    if r.get('is_correct', False):
      type_stats[qt]['correct'] += 1

  cond_stats: dict[str, dict[str, int]] = {}
  for r in results:
    cond = r.get('condition', 'unknown')
    cond_stats.setdefault(cond, {'correct': 0, 'total': 0})
    cond_stats[cond]['total'] += 1
    if r.get('is_correct', False):
      cond_stats[cond]['correct'] += 1

  return {
      'model': model_name,
      'include_audio': include_audio,
      'total_items': total_count,
      'correct': correct_count,
      'accuracy': accuracy,
      'random_baseline': random_baseline,
      'by_question_type': {
          qt: {
              'correct': s['correct'],
              'total': s['total'],
              'accuracy': s['correct'] / s['total'] if s['total'] else 0,
          }
          for qt, s in type_stats.items()
      },
      'by_condition': {
          c: {
              'correct': s['correct'],
              'total': s['total'],
              'accuracy': s['correct'] / s['total'] if s['total'] else 0,
          }
          for c, s in cond_stats.items()
      },
      'results': results,
  }


def _print_evaluation_summary(
    summary: dict[str, Any],
    text_only: bool,
    audio_only: bool,
    include_audio: bool,
) -> None:
  """Prints formatted terminal evaluation summary and error reports."""
  if text_only:
    modality = 'text-only'
  elif audio_only:
    modality = 'audio-only'
  elif include_audio:
    modality = 'video+audio'
  else:
    modality = 'video-only'
  logging.info('\n%s', '=' * 60)
  logging.info('MCQ EVALUATION — %s', summary['model'])
  logging.info('Modality: %s', modality)
  logging.info('=' * 60)
  logging.info(
      'Overall accuracy: %d/%d (%.1f%%)',
      summary['correct'],
      summary['total_items'],
      summary['accuracy'] * 100,
  )
  logging.info('Random baseline:  %.1f%%', summary['random_baseline'] * 100)
  logging.info('\nBy question type:')
  for qt, s in sorted(summary['by_question_type'].items()):
    logging.info(
        '  %12s: %d/%d (%.0f%%)',
        qt,
        s['correct'],
        s['total'],
        s['accuracy'] * 100,
    )
  logging.info('\nBy condition:')
  for c, s in sorted(summary['by_condition'].items()):
    logging.info(
        '  %-30s: %d/%d (%.0f%%)',
        c[:30],
        s['correct'],
        s['total'],
        s['accuracy'] * 100,
    )
  logging.info('')


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> None:
  """Runs MCQ evaluation against ClinicalSense benchmark items."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  items = _load_benchmark_items(_BENCHMARK_FILE.value)
  results, evaluated_keys = _load_or_resume_checkpoint(_OUTPUT_FILE.value)

  lm = language_model.LanguageModel(model_name=_MODEL_NAME.value)
  logging.info('Initializing LM with %s...', _MODEL_NAME.value)

  results = _run_evaluation_loop(
      items=items,
      evaluated_keys=evaluated_keys,
      results=results,
      lm=lm,
      clips_dir=_CLIPS_DIR.value,
      output_file=_OUTPUT_FILE.value,
      model_name=_MODEL_NAME.value,
      include_audio=_INCLUDE_AUDIO.value,
      text_only=_TEXT_ONLY.value,
      audio_only=_AUDIO_ONLY.value,
      fps=_FPS.value,
  )

  summary = _compute_summary_stats(
      results=results,
      model_name=_MODEL_NAME.value,
      include_audio=_INCLUDE_AUDIO.value,
  )

  with open(_OUTPUT_FILE.value, 'w') as f:
    json.dump(summary, f, indent=2)
  logging.info('Results saved to %s', _OUTPUT_FILE.value)

  _print_evaluation_summary(
      summary=summary,
      text_only=_TEXT_ONLY.value,
      audio_only=_AUDIO_ONLY.value,
      include_audio=_INCLUDE_AUDIO.value,
  )


if __name__ == '__main__':
  app.run(main)
