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
r"""Stage 1: Visual Evidence Extraction from consultation video clips.

Two-pass pipeline:
  1. BLIND observation — extract visual evidence with no annotation context
  2. TRANSCRIPT VERIFICATION — transcript verification across 5 EIM facets

Supports two clip sources:
  - Indicator-based clips from --clips_dir (e.g. 10.0_indicator_0_window_0.mp4)
    with case JSON files and clinician annotations.
  - Clips listed in a Stage 0 manifest (--stage0_manifest), written by
    stage0_discover.py, which also carries each clip's transcript.

Usage:
  python stage1_prototype.py \
    --clips_dir=/path/to/clips/ \
    --cases_dir=/path/to/cases/ \
    --case_annotations_csv=/path/to/case_annotations.csv \
    --output_file=/tmp/stage1_visual_evidence.json

Usage (Stage 0 clips):
  python stage1_prototype.py \
    --stage0_manifest=/tmp/stage0_manifest.json \
    --output_file=/tmp/stage1_visual_evidence.json \
    --max_items=0
"""

import csv
import functools
import json
import os
import subprocess
import traceback
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import pydantic

import constraints
import content_utils
import data_models
import language_model
import template_utils

_CLIPS_DIR = flags.DEFINE_string(
    'clips_dir',
    '',  # directory containing local clips
    'Directory containing video clips.',
)
_CASES_DIR = flags.DEFINE_string(
    'cases_dir',
    '',  # directory containing local cases
    'Directory containing JSON AvEvaluationCase files.',
)
_CASE_ANNOTATIONS_CSV = flags.DEFINE_string(
    'case_annotations_csv',
    '',
    'Path to the clinician case annotations CSV. Expected columns: Case ID,'
    ' Condition, Patient Gender, Key Medical Information, Key Audio Video'
    ' Indicator, Case Notes.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    language_model.DEFAULT_MODEL,
    'Gemini model ID to use for extraction.',
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    '/tmp/stage1_visual_evidence.json',
    'Path to write the visual evidence output JSON.',
)
_MAX_ITEMS = flags.DEFINE_integer(
    'max_items',
    10,
    'Maximum number of clips to process (0 = all).',
)
_CLIPS_LIST_FILE = flags.DEFINE_string(
    'clips_list_file',
    '',
    'Path to text file containing clip filenames (one per line).',
)
_STAGE0_MANIFEST = flags.DEFINE_string(
    'stage0_manifest',
    '',
    'Path to a Stage 0 manifest JSON written by stage0_discover.py. When set,'
    ' clips and their transcripts are read from the manifest instead of'
    ' --clips_dir / --clips_list_file, and --cases_dir /'
    ' --case_annotations_csv are not used.',
)
_FPS = flags.DEFINE_float(
    'fps',
    5.0,
    'Frames per second for Gemini video decoding (default: 5.0 FPS).',
)

_FACET_KEYS = ('action', 'spatial', 'temporal', 'appearance', 'absence')
_DEFAULT_CLIP_DURATION_SEC = 30.0
_CLIP_EXTENSIONS = ('.mp4', '.mov', '.webm', '.mkv')
# Condition used for Stage 0 clips, which have no case annotations.
_DEFAULT_STAGE0_CONDITION = 'General Clinical'


# ---------------------------------------------------------------------------
# Facet Definitions
# ---------------------------------------------------------------------------


def _load_eim_facets() -> dict[str, dict[str, str]]:
  """Loads EIM facet definitions from standalone JSON file."""
  facets_path = os.path.join(os.path.dirname(__file__), 'eim_facets.json')
  with open(facets_path, 'r') as f:
    return json.load(f)


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

_BLIND_OBS_FIELDS = {k: f'{k}_observations' for k in _FACET_KEYS}


@pydantic.dataclasses.dataclass
class BlindObservation(constraints.BaseModel):
  """Unbiased visual observations organized by EIM perception facets."""

  action_observations: str = pydantic.Field(
      description='Goal-directed physical behaviors and intentional actions.'
  )
  spatial_observations: str = pydantic.Field(
      description='Precise anatomical regions, laterality, and body positions.'
  )
  temporal_observations: str = pydantic.Field(
      description='Timeline of discrete observable events with timestamps.'
  )
  appearance_observations: str = pydantic.Field(
      description='Visual physical exam attributes (distress, eyes, posture).'
  )
  absence_observations: str = pydantic.Field(
      description='Clinically expected findings that are visually absent.'
  )
  primary_sign: str = pydantic.Field(
      description='The single most prominent clinical sign visible in clip.'
  )


@pydantic.dataclasses.dataclass
class FacetVerification(constraints.BaseModel):
  """Transcript verification verdict for a single EIM facet observation."""

  eim_facet: str = pydantic.Field(description='Primary EIM facet name.')
  transcript_verdict: str = pydantic.Field(
      description='Verdict: SUPPORTED, NOT_MENTIONED, or CONTRADICTED.'
  )
  transcript_evidence: str = pydantic.Field(
      description='Specific transcript quote or explanation. Max 60 words.'
  )


@pydantic.dataclasses.dataclass
class FacetVerificationBatch(constraints.BaseModel):
  """Transcript verification verdicts across all 5 EIM facets."""

  facet_verdicts: list[FacetVerification] = pydantic.Field(
      description=(
          'Exactly 5 verification entries, one per EIM perception facet.'
      )
  )


# ---------------------------------------------------------------------------
# Data loaders & clip utilities
# ---------------------------------------------------------------------------


def _load_case(
    case_id: str, cases_dir: str
) -> data_models.AvEvaluationCase | None:
  """Loads an AvEvaluationCase from JSON, or returns None if not found."""
  case_file = os.path.join(cases_dir, f'{case_id}.json')
  if not os.path.exists(case_file):
    return None
  with open(case_file, 'r') as f:
    return data_models.AvEvaluationCase(**json.load(f))


def _load_case_annotations(csv_path: str) -> dict[str, dict[str, str]]:
  """Loads case annotations CSV into a dict keyed by case ID."""
  annotations = {}
  with open(csv_path, 'r') as f:
    for row in csv.DictReader(f):
      case_id = row.get('Case ID', '').strip()
      if case_id:
        annotations[case_id] = {
            'condition': row.get('Condition', '').strip(),
            'patient_gender': row.get('Patient Gender', '').strip(),
            'key_medical_information': (
                row.get('Key Medical Information', '').strip()
            ),
            'key_av_indicator': (
                row.get('Key Audio Video Indicator', '').strip()
            ),
            'case_notes': row.get('Case Notes', '').strip(),
        }
  return annotations


def _get_transcript_segment(
    case: data_models.AvEvaluationCase,
    indicator_idx: int,
    padding_s: float = 15.0,
) -> str:
  """Extracts the transcript segment overlapping an indicator's time window.

  Args:
    case: The loaded AvEvaluationCase object.
    indicator_idx: Index of the AV indicator.
    padding_s: Seconds of padding before and after the clip window to capture
      clinician comments made shortly after observing a sign.

  Returns:
    Formatted transcript lines overlapping the padded window.
  """
  indicators = case.audio_video_indicators.av_indicators
  if indicator_idx >= len(indicators):
    return ''

  indicator = indicators[indicator_idx]
  if indicator.is_continuous or not indicator.time_windows:
    windows = [(0.0, float('inf'))]
  else:
    windows = indicator.time_windows

  overlapping_turns = []
  for turn in case.transcript.turns:
    for win_start, win_end in windows:
      padded_start = max(0, win_start - padding_s)
      padded_end = win_end + padding_s
      if turn.start_time < padded_end and turn.end_time > padded_start:
        overlapping_turns.append(turn)
        break

  if not overlapping_turns:
    return ''

  return '\n'.join(
      f'[{t.start_time:.1f}-{t.end_time:.1f}] {t.speaker}: {t.text}'
      for t in overlapping_turns
  )


def _get_symptom_string(
    case: data_models.AvEvaluationCase,
    indicator_idx: int,
) -> str:
  """Gets the symptom string for a specific indicator."""
  indicators = case.audio_video_indicators.av_indicators
  if indicator_idx >= len(indicators):
    return ''
  return indicators[indicator_idx].av_indicator


def _parse_clip_name(clip_name: str) -> tuple[str, int] | None:
  """Parses '10.0_indicator_0_window_0.mp4' -> ('10.0', 0). None on failure."""
  parts = clip_name.replace('.mp4', '').split('_indicator_')
  if len(parts) != 2:
    return None
  try:
    return parts[0], int(parts[1].split('_')[0])
  except (ValueError, IndexError):
    return None


def _get_clip_duration_sec(clip_path: str) -> float:
  """Returns the clip duration in seconds, read with ffprobe.

  Falls back to _DEFAULT_CLIP_DURATION_SEC if the duration cannot be read.

  Args:
    clip_path: Path to a local video clip.
  """
  cmd = [
      'ffprobe',
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      clip_path,
  ]
  try:
    output = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout
    return float(output.strip())
  except (OSError, subprocess.CalledProcessError, ValueError) as e:
    logging.warning('Failed to get clip duration for %s: %s', clip_path, e)
  return _DEFAULT_CLIP_DURATION_SEC


# ---------------------------------------------------------------------------
# Pipeline steps & helpers
# ---------------------------------------------------------------------------


def _load_dataset_sources(
    clips_dir: str,
    annotations_csv: str,
    clips_list_file: str,
    max_items: int,
) -> tuple[list[str], dict[str, dict[str, str]]]:
  """Loads candidate target clips and clinical case annotations."""
  logging.info('Loading case annotations...')
  case_annotations = _load_case_annotations(annotations_csv)
  logging.info('Loaded annotations for %d cases.', len(case_annotations))

  if clips_list_file and os.path.exists(clips_list_file):
    logging.info('Loading target clips from list file: %s', clips_list_file)
    with open(clips_list_file, 'r') as f:
      target_clips = [line.strip() for line in f if line.strip()]
    logging.info('Loaded %d clips from list file.', len(target_clips))
  else:
    logging.info('Loading target clips from clips directory: %s', clips_dir)
    target_clips = sorted(
        name
        for name in os.listdir(clips_dir)
        if name.lower().endswith(_CLIP_EXTENSIONS)
    )
    logging.info('Found %d clips.', len(target_clips))

  if max_items > 0:
    target_clips = target_clips[:max_items]

  return target_clips, case_annotations


def _process_single_clip(
    clip_name: str,
    case_annotations: dict[str, dict[str, str]],
    cases_dir: str,
    clips_dir: str,
    lm: language_model.LanguageModel,
    eim_facets: dict[str, dict[str, str]],
    fps: float | None = None,
) -> dict[str, Any] | None:
  """Runs 2-pass evidence extraction for a single clinical video clip."""
  parsed = _parse_clip_name(clip_name)
  if not parsed:
    logging.warning('Cannot parse clip name: %s', clip_name)
    return None
  case_id, indicator_idx = parsed
  case_id_int = str(int(float(case_id)))

  case = _load_case(case_id, cases_dir)
  if not case:
    logging.warning('Case not found for %s', clip_name)
    return None

  case_ann = case_annotations.get(case_id_int, {})
  symptom = _get_symptom_string(case, indicator_idx)
  transcript = _get_transcript_segment(case, indicator_idx)
  transcript_or_none = transcript or 'No transcript available.'

  noaudio_name = clip_name.replace('.mp4', '_noaudio.mp4')
  noaudio_path = os.path.join(clips_dir, noaudio_name)
  clip_path = (
      noaudio_path
      if os.path.exists(noaudio_path)
      else os.path.join(clips_dir, clip_name)
  )

  if not os.path.exists(clip_path):
    logging.warning('Clip not found: %s', clip_path)
    return None

  duration_sec = _get_clip_duration_sec(clip_path)
  return _run_two_pass_extraction(
      lm=lm,
      eim_facets=eim_facets,
      clip_path=clip_path,
      clip_name=clip_name,
      case_id=case_id,
      indicator_idx=indicator_idx,
      condition=case_ann.get('condition', ''),
      symptom=symptom,
      transcript=transcript_or_none,
      duration_sec=duration_sec,
      fps=fps,
  )


def _load_stage0_clips(manifest_path: str) -> list[dict[str, Any]]:
  """Loads the clip entries of a Stage 0 manifest ({'clips': [...]} or list)."""
  with open(manifest_path, 'r') as f:
    data = json.load(f)
  clips = data if isinstance(data, list) else data.get('clips', [])
  logging.info(
      'Loaded %d clips from Stage 0 manifest: %s', len(clips), manifest_path
  )
  return clips


def _stage0_clip_name(clip_entry: dict[str, Any]) -> str:
  """Returns the clip file name used as the result / checkpoint key."""
  return clip_entry.get('clip_file') or os.path.basename(
      clip_entry.get('clip_path', '')
  )


def _process_stage0_clip(
    clip_entry: dict[str, Any],
    lm: language_model.LanguageModel,
    eim_facets: dict[str, dict[str, str]],
    fps: float | None = None,
) -> dict[str, Any] | None:
  """Runs 2-pass evidence extraction for a single Stage 0 manifest clip."""
  clip_name = _stage0_clip_name(clip_entry)
  noaudio_path = clip_entry.get('clip_path_noaudio', '')
  av_path = clip_entry.get('clip_path', '')
  if noaudio_path and os.path.exists(noaudio_path):
    clip_path = noaudio_path
  elif av_path and os.path.exists(av_path):
    clip_path = av_path
  else:
    logging.warning('Clip not found: %s / %s', av_path, noaudio_path)
    return None

  duration_sec = clip_entry.get('clip_duration_sec') or _get_clip_duration_sec(
      clip_path
  )
  case_id = clip_entry.get('case_id') or clip_entry.get('source_video_id', '')
  return _run_two_pass_extraction(
      lm=lm,
      eim_facets=eim_facets,
      clip_path=clip_path,
      clip_name=clip_name,
      case_id=case_id,
      indicator_idx=None,
      condition=clip_entry.get('condition') or _DEFAULT_STAGE0_CONDITION,
      symptom=clip_entry.get('symptom')
      or clip_entry.get('clinical_event_description', ''),
      transcript=clip_entry.get('transcript') or 'No transcript available.',
      duration_sec=float(duration_sec),
      stage0_metadata={
          'source_video_id': clip_entry.get('source_video_id', ''),
          'discovery_mode': clip_entry.get('discovery_mode', ''),
          'primary_eim_facet': clip_entry.get('primary_eim_facet', ''),
          'clinical_event_description': clip_entry.get(
              'clinical_event_description', ''
          ),
          'start_sec': clip_entry.get('start_sec', 0.0),
          'end_sec': clip_entry.get('end_sec', 0.0),
      },
      fps=fps,
  )


def _run_two_pass_extraction(
    lm: language_model.LanguageModel,
    eim_facets: dict[str, dict[str, str]],
    clip_path: str,
    clip_name: str,
    case_id: str,
    indicator_idx: int | None,
    condition: str,
    symptom: str,
    transcript: str,
    duration_sec: float,
    stage0_metadata: dict[str, Any] | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
  """Core two-pass pipeline: blind observation + transcript verification.

  Args:
    lm: Language model instance.
    eim_facets: EIM facet definitions.
    clip_path: Path of the clip sent to the model (muted if available).
    clip_name: Clip file name, used as the result / checkpoint key.
    case_id: Case (or source video) identifier.
    indicator_idx: Indicator index for indicator-based clips, else None.
    condition: Clinical condition string ('' if unknown).
    symptom: Symptom / indicator description.
    transcript: Transcript segment text, or 'No transcript available.'.
    duration_sec: Clip duration in seconds.
    stage0_metadata: Stage 0 metadata to attach to the result, if any.
    fps: Frame rate at which Gemini samples the clip. If None, Gemini uses its
      default (1 FPS).

  Returns:
    The result dict for the clip, with status 'success' or 'error'.
  """
  m, s = divmod(int(duration_sec), 60)
  clip_duration_str = f'{m:02d}:{s:02d}'

  try:
    # ===== PASS 1: BLIND OBSERVATION =====
    pass1_prompt = template_utils.render_template(
        'stage1_pass1_blind',
        facets=eim_facets,
    )
    pass1_prompt += constraints.get_constraint_suffix(cls=BlindObservation)

    with content_utils.media_part(lm.client, clip_path, fps=fps) as clip_part:
      blind_obs, _ = lm.generate_thought_and_object(
          [clip_part, pass1_prompt], cls=BlindObservation
      )

    # ===== PASS 2: FACET-LEVEL TRANSCRIPT VERIFICATION =====
    observations = [
        {
            'label': eim_facets[k]['label'],
            'value': getattr(blind_obs, _BLIND_OBS_FIELDS[k]),
        }
        for k in _FACET_KEYS
    ]
    pass2_prompt = template_utils.render_template(
        'stage1_pass2_grounded',
        primary_sign=blind_obs.primary_sign,
        observations=observations,
        condition=condition or 'Unknown',
        symptom=symptom or 'Not specified',
        transcript=transcript,
    )
    pass2_prompt += constraints.get_constraint_suffix(
        cls=FacetVerificationBatch
    )

    verified_batch, _ = lm.generate_thought_and_object(
        pass2_prompt, cls=FacetVerificationBatch
    )

    verdicts = {
        fv.eim_facet.upper(): {
            'verdict': fv.transcript_verdict,
            'evidence': fv.transcript_evidence,
        }
        for fv in verified_batch.facet_verdicts
    }

    v_counts = {'SUPPORTED': 0, 'NOT_MENTIONED': 0, 'CONTRADICTED': 0}
    for fv in verified_batch.facet_verdicts:
      verdict_str = fv.transcript_verdict.upper()
      if verdict_str in v_counts:
        v_counts[verdict_str] += 1
      else:
        v_counts['NOT_MENTIONED'] += 1

    result = {
        'clip': clip_name,
        'case_id': case_id,
        'indicator_idx': indicator_idx,
        'condition': condition,
        'symptom': symptom,
        'clip_duration_sec': round(duration_sec, 2),
        'clip_duration_str': clip_duration_str,
        'evidence': {
            **{
                k: getattr(blind_obs, _BLIND_OBS_FIELDS[k]) for k in _FACET_KEYS
            },
            'primary_sign': blind_obs.primary_sign,
        },
        'facet_verdicts': verdicts,
        'verdict_summary': v_counts,
        'status': 'success',
    }

  except Exception as e:  # pylint: disable=broad-except
    logging.error('Error processing %s: %s', clip_name, e)
    logging.error(traceback.format_exc())
    result = {
        'clip': clip_name,
        'case_id': case_id,
        'indicator_idx': indicator_idx,
        'condition': condition,
        'symptom': symptom,
        'evidence': None,
        'facet_verdicts': None,
        'verdict_summary': None,
        'status': 'error',
        'error': str(e),
    }
  if stage0_metadata is not None:
    result['stage0'] = stage0_metadata
  return result


def _run_extraction_pipeline(
    clips: list[str],
    case_annotations: dict[str, dict[str, str]],
    cases_dir: str,
    clips_dir: str,
    lm: language_model.LanguageModel,
    eim_facets: dict[str, dict[str, str]],
    output_file: str,
    model_name: str,
    stage0_clips: list[dict[str, Any]] | None = None,
    fps: float | None = None,
) -> list[dict[str, Any]]:
  """Runs the 2-pass extraction pipeline across clips with incremental checkpoints.

  Args:
    clips: Indicator-based clip file names (ignored if stage0_clips is set).
    case_annotations: Clinician annotations keyed by case ID.
    cases_dir: Directory containing case JSON files.
    clips_dir: Directory containing the indicator-based clips.
    lm: Language model instance.
    eim_facets: EIM facet definitions.
    output_file: Path of the incremental checkpoint / output JSON.
    model_name: Model name recorded in the output.
    stage0_clips: Clip entries of a Stage 0 manifest, or None.
    fps: Frame rate at which Gemini samples each clip. If None, Gemini uses its
      default (1 FPS).

  Returns:
    The extraction results, including ones resumed from the checkpoint.
  """
  results: list[dict[str, Any]] = []
  processed_clips = set()

  if os.path.exists(output_file):
    try:
      with open(output_file, 'r') as f:
        existing_data = json.load(f)
        results = existing_data.get('results', [])
        processed_clips = {r['clip'] for r in results if 'clip' in r}
      logging.info(
          'Found existing checkpoint with %d processed clips.',
          len(processed_clips),
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('Could not load checkpoint: %s', e)

  if stage0_clips is not None:
    source = 'stage0_manifest'
    work_items = [
        (
            _stage0_clip_name(entry),
            functools.partial(
                _process_stage0_clip, entry, lm, eim_facets, fps=fps
            ),
        )
        for entry in stage0_clips
    ]
  else:
    source = 'clips_dir'
    work_items = [
        (
            clip_name,
            functools.partial(
                _process_single_clip,
                clip_name=clip_name,
                case_annotations=case_annotations,
                cases_dir=cases_dir,
                clips_dir=clips_dir,
                lm=lm,
                eim_facets=eim_facets,
                fps=fps,
            ),
        )
        for clip_name in clips
    ]

  for i, (clip_name, process_clip) in enumerate(work_items):
    if clip_name in processed_clips:
      logging.info(
          '[%d/%d] Skipping %s (already in checkpoint).',
          i + 1,
          len(work_items),
          clip_name,
      )
      continue

    logging.info('[%d/%d] Processing %s...', i + 1, len(work_items), clip_name)
    res = process_clip()
    if res is not None:
      results.append(res)
      _save_evidence_checkpoint(results, output_file, model_name, source)

  return results


def _save_evidence_checkpoint(
    results: list[dict[str, Any]],
    output_file: str,
    model_name: str,
    source: str = 'clips_dir',
) -> None:
  """Saves incremental checkpoint to output_file."""
  summary = {
      'model': model_name,
      'pipeline': 'two_pass_blind_facet_transcript_verify',
      'source': source,
      'num_processed': len(results),
      'num_success': sum(1 for r in results if r.get('status') == 'success'),
      'num_error': sum(1 for r in results if r.get('status') == 'error'),
      'results': results,
  }
  with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)


def _print_extraction_summary(
    results: list[dict[str, Any]],
    model_name: str,
) -> None:
  """Prints terminal summary of extracted visual evidence."""
  num_processed = len(results)
  num_success = sum(1 for r in results if r.get('status') == 'success')
  num_error = sum(1 for r in results if r.get('status') == 'error')

  logging.info('\n%s', '=' * 60)
  logging.info('STAGE 1: TWO-PASS VISUAL EVIDENCE EXTRACTION — SUMMARY')
  logging.info('%s', '=' * 60)
  logging.info('Model: %s', model_name)
  logging.info(
      'Processed: %d  |  Success: %d  |  Errors: %d',
      num_processed,
      num_success,
      num_error,
  )

  for r in results:
    if r.get('status') == 'success' and r.get('evidence'):
      v_sum = r.get('verdict_summary')
      v_dict = v_sum if isinstance(v_sum, dict) else {}
      logging.info('\n--- %s (%s) ---', r['clip'], r.get('condition', ''))
      logging.info('  Primary sign: %s', r['evidence']['primary_sign'][:120])
      logging.info(
          '  Facet verdicts: %d supported, %d not_mentioned, %d contradicted',
          v_dict.get('SUPPORTED', 0),
          v_dict.get('NOT_MENTIONED', 0),
          v_dict.get('CONTRADICTED', 0),
      )


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> None:
  """Runs the two-pass visual evidence extraction pipeline."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  eim_facets = _load_eim_facets()
  lm = language_model.LanguageModel(model_name=_MODEL_NAME.value)

  if _STAGE0_MANIFEST.value:
    stage0_clips = _load_stage0_clips(_STAGE0_MANIFEST.value)
    if _MAX_ITEMS.value > 0:
      stage0_clips = stage0_clips[: _MAX_ITEMS.value]
    results = _run_extraction_pipeline(
        clips=[],
        case_annotations={},
        cases_dir='',
        clips_dir='',
        lm=lm,
        eim_facets=eim_facets,
        output_file=_OUTPUT_FILE.value,
        model_name=_MODEL_NAME.value,
        stage0_clips=stage0_clips,
        fps=_FPS.value,
    )
    _print_extraction_summary(results, _MODEL_NAME.value)
    return

  clips, case_annotations = _load_dataset_sources(
      clips_dir=_CLIPS_DIR.value,
      annotations_csv=_CASE_ANNOTATIONS_CSV.value,
      clips_list_file=_CLIPS_LIST_FILE.value,
      max_items=_MAX_ITEMS.value,
  )

  results = _run_extraction_pipeline(
      clips=clips,
      case_annotations=case_annotations,
      cases_dir=_CASES_DIR.value,
      clips_dir=_CLIPS_DIR.value,
      lm=lm,
      eim_facets=eim_facets,
      output_file=_OUTPUT_FILE.value,
      model_name=_MODEL_NAME.value,
      fps=_FPS.value,
  )

  _print_extraction_summary(results, _MODEL_NAME.value)


if __name__ == '__main__':
  app.run(main)
