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
r"""Stage 0: Clinical event discovery and clip extraction from full videos.

Scans full-length clinical recordings, such as simulated telehealth
consultations (with a timestamped transcript) or narrated instructional videos
such as MedVidQA (speech only in the audio track), for clinically relevant
visual events, and cuts one short clip per event for the downstream stages.

Discovery modes (--discovery_mode):

  coarse_to_fine (default), the scan procedure described in the paper:
    Pass 1A (language cues): a text-only pass over the timestamped transcript
      proposes candidate windows with EIM facet labels and trigger quotes. If
      no transcript is provided, the speech is first transcribed from the
      audio track (--transcribe_audio).
    Pass 1B (visual cues): the muted full video is scanned at --coarse_fps
      (default 5 FPS) to propose candidate windows.
    Fusion: the union of both sets is sorted, and a window is merged into the
      previous one when it starts at most --merge_gap_sec after the previous
      window ends and the two share an EIM facet.
    Pass 2 (fine localization): each merged window, padded by
      --window_padding_sec, is cut as a muted clip and re-scanned at
      --fine_fps (default 10 FPS), with the transcript excerpt as reference
      text, to localize individual events. Events that repeat an earlier
      event are then dropped (--dedup_scope).

  single_pass: one multimodal call on the full video with audio at
    --single_pass_fps (default 5 FPS) that lists events and summarizes their
    speech. This mode was used to build the released MedVidQA benchmark (v1).

For every event, <clip>.mp4 (video and audio), <clip>_noaudio.mp4 and
<clip>_audio.wav (16 kHz mono) are written to --output_clips_dir, and the
clip is recorded in a JSON manifest that stage1_prototype.py reads with
--stage0_manifest. Re-running with the same --output_manifest skips videos
that were already processed successfully. Requires ffmpeg and ffprobe.

Usage:
  python stage0_discover.py \
    --videos_dir=/path/to/videos/ \
    --transcripts_dir=/path/to/transcripts/ \
    --output_clips_dir=/tmp/stage0_clips/ \
    --output_manifest=/tmp/stage0_manifest.json
"""

import collections
from concurrent import futures
import contextlib
import dataclasses
import json
import os
import re
import subprocess
import tempfile
import threading
from typing import Any, Iterable, Iterator, Sequence

from absl import app
from absl import flags
from absl import logging
import pydantic

import constraints
import content_utils
import language_model
import template_utils

COARSE_TO_FINE = 'coarse_to_fine'
SINGLE_PASS = 'single_pass'

TRANSCRIPT_SOURCE = 'TRANSCRIPT'
VISUAL_SOURCE = 'VISUAL'

# Per-mode defaults for --padding_sec and --max_clip_duration_sec, used only
# when the flag is not set. A maximum of 0 means no maximum.
_MODE_DEFAULTS = {
    COARSE_TO_FINE: {'padding_sec': 5.0, 'max_clip_duration_sec': 60.0},
    SINGLE_PASS: {'padding_sec': 2.0, 'max_clip_duration_sec': 0.0},
}

_VIDEO_MIMETYPE = 'video/mp4'
_AUDIO_MIMETYPE = 'audio/wav'
_MIN_OUTPUT_BYTES = 1024
_MIN_SEGMENT_SEC = 0.5
_MAX_GEMINI_FPS = 24.0
_NO_TRANSCRIPT = 'No transcript available.'
_NO_SPEECH = 'No speech detected.'

_VIDEOS_DIR = flags.DEFINE_string(
    'videos_dir',
    '',
    'Directory containing the full-length source .mp4 videos.',
)
_VIDEOS_LIST_FILE = flags.DEFINE_string(
    'videos_list_file',
    '',
    'Text file with one local video path per line. Takes precedence over'
    ' --videos_dir.',
)
_TRANSCRIPTS_DIR = flags.DEFINE_string(
    'transcripts_dir',
    '',
    'Optional directory of <video_id>.json transcripts, each a JSON list of'
    ' {"start_sec", "end_sec", "speaker" (optional), "text"} segments, where'
    ' <video_id> is the video file name without its extension.',
)
_OUTPUT_CLIPS_DIR = flags.DEFINE_string(
    'output_clips_dir', None, 'Directory to write the extracted clips to.'
)
_OUTPUT_MANIFEST = flags.DEFINE_string(
    'output_manifest', None, 'Path to write the Stage 0 manifest JSON to.'
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', language_model.DEFAULT_MODEL, 'Gemini model ID.'
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 4, 'Number of videos to process in parallel.'
)
_MAX_VIDEOS = flags.DEFINE_integer(
    'max_videos', 0, 'Maximum number of videos to process (0 = all).'
)
_DISCOVERY_MODE = flags.DEFINE_enum(
    'discovery_mode',
    COARSE_TO_FINE,
    [COARSE_TO_FINE, SINGLE_PASS],
    'coarse_to_fine: transcript and visual coarse scans, window fusion and'
    ' per-window fine localization (the procedure described in the paper).'
    ' single_pass: one call on the full video with audio, the procedure used'
    ' to build the released MedVidQA benchmark (v1).',
)
_TRANSCRIBE_AUDIO = flags.DEFINE_bool(
    'transcribe_audio',
    True,
    'coarse_to_fine only: when a video has no transcript in'
    ' --transcripts_dir, transcribe its audio track with Gemini to drive the'
    ' language scan. If False, such videos only get the visual scan.',
)
_COARSE_FPS = flags.DEFINE_float(
    'coarse_fps', 5.0, 'Frame rate of the coarse visual scan (Pass 1B).'
)
_FINE_FPS = flags.DEFINE_float(
    'fine_fps', 10.0, 'Frame rate of the per-window fine scan (Pass 2).'
)
_SINGLE_PASS_FPS = flags.DEFINE_float(
    'single_pass_fps', 5.0, 'Frame rate of the single_pass discovery call.'
)
_MERGE_GAP_SEC = flags.DEFINE_float(
    'merge_gap_sec',
    10.0,
    'A candidate window is merged into the previous one when it starts at'
    ' most this many seconds after the previous window ends and the two share'
    ' an EIM facet.',
)
_WINDOW_PADDING_SEC = flags.DEFINE_float(
    'window_padding_sec',
    5.0,
    'Padding added to each side of a merged window before the fine scan.',
)
_TRANSCRIPT_PADDING_SEC = flags.DEFINE_float(
    'transcript_padding_sec',
    5.0,
    'Transcript context, in seconds, added to each side of a fine-scan'
    ' segment when selecting its transcript excerpt.',
)
_DEDUP_SCOPE = flags.DEFINE_enum(
    'dedup_scope',
    'video',
    ['video', 'window'],
    'video: drop any event that repeats an earlier event of the same video'
    ' (same primary facet, spatial target, appearance state, action verb and'
    ' pertinent-negative flag). window: only drop a repeat that starts at'
    ' most --dedup_max_gap_sec after the earlier event ends.',
)
_DEDUP_MAX_GAP_SEC = flags.DEFINE_float(
    'dedup_max_gap_sec', 120.0, 'Maximum gap for --dedup_scope=window.'
)
_PADDING_SEC = flags.DEFINE_float(
    'padding_sec',
    None,
    'Padding added to each side of an event when cutting its clip. Default:'
    ' 5.0 for coarse_to_fine, 2.0 for single_pass.',
)
_MIN_CLIP_DURATION_SEC = flags.DEFINE_float(
    'min_clip_duration_sec',
    5.0,
    'Clips shorter than this (after padding) are skipped.',
)
_MAX_CLIP_DURATION_SEC = flags.DEFINE_float(
    'max_clip_duration_sec',
    None,
    'Clips longer than this (after padding) are truncated; 0 means no'
    ' maximum. Default: 60.0 for coarse_to_fine, 0 for single_pass.',
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class TranscriptSegment(constraints.BaseModel):
  """One timestamped speech segment."""

  start_sec: float = pydantic.Field(
      description='Start time of the segment in seconds.'
  )
  end_sec: float = pydantic.Field(
      description='End time of the segment in seconds.'
  )
  text: str = pydantic.Field(description='The words spoken, verbatim.')
  speaker: str = pydantic.Field(
      default='',
      description=(
          'Short speaker label (e.g. "Doctor", "Patient", "Narrator"), or an'
          ' empty string if unknown.'
      ),
  )


@pydantic.dataclasses.dataclass
class TranscriptionResult(constraints.BaseModel):
  """Timestamped transcript of a video's audio track."""

  segments: list[TranscriptSegment] = pydantic.Field(
      description=(
          'All speech segments in chronological order. Empty if there is no'
          ' intelligible speech.'
      )
  )


@pydantic.dataclasses.dataclass
class LanguageCueWindow(constraints.BaseModel):
  """A candidate window proposed from the transcript (Pass 1A)."""

  start_sec: float = pydantic.Field(
      description='Approximate start timestamp in seconds.'
  )
  end_sec: float = pydantic.Field(
      description='Approximate end timestamp in seconds.'
  )
  description: str = pydantic.Field(
      description='What clinical activity the speech suggests.'
  )
  suggested_eim_facets: list[str] = pydantic.Field(
      description=(
          'EIM facets likely relevant based on the speech. One or more of:'
          ' ACTION, SPATIAL, TEMPORAL, APPEARANCE, ABSENCE.'
      )
  )
  trigger_quote: str = pydantic.Field(
      description='The utterance that triggered this proposal.'
  )


@pydantic.dataclasses.dataclass
class LanguageScanResult(constraints.BaseModel):
  """Pass 1A output: candidate windows from the transcript."""

  candidate_windows: list[LanguageCueWindow] = pydantic.Field(
      description='All candidate windows found in the transcript. May be empty.'
  )


@pydantic.dataclasses.dataclass
class VisualCueWindow(constraints.BaseModel):
  """A candidate window proposed from the muted video (Pass 1B)."""

  start_sec: float = pydantic.Field(
      description='Approximate start timestamp in seconds.'
  )
  end_sec: float = pydantic.Field(
      description='Approximate end timestamp in seconds.'
  )
  description: str = pydantic.Field(
      description='Brief description of the physical activity detected.'
  )
  suggested_eim_facets: list[str] = pydantic.Field(
      description=(
          'EIM facets likely relevant to this window. One or more of:'
          ' ACTION, SPATIAL, TEMPORAL, APPEARANCE, ABSENCE.'
      )
  )


@pydantic.dataclasses.dataclass
class VisualScanResult(constraints.BaseModel):
  """Pass 1B output: candidate windows from the coarse visual scan."""

  candidate_windows: list[VisualCueWindow] = pydantic.Field(
      description=(
          'All windows with clinically relevant physical activity. May be'
          ' empty.'
      )
  )


@pydantic.dataclasses.dataclass
class EIMEventCandidate(constraints.BaseModel):
  """A single clinically relevant visual event from the fine scan (Pass 2)."""

  start_sec: float = pydantic.Field(
      description=(
          'Precise start timestamp of the event in seconds, relative to the'
          ' start of the segment.'
      )
  )
  end_sec: float = pydantic.Field(
      description=(
          'Precise end timestamp of the event in seconds, relative to the'
          ' start of the segment.'
      )
  )
  primary_eim_facet: str = pydantic.Field(
      description=(
          'Primary EIM facet: ACTION, SPATIAL, TEMPORAL, APPEARANCE, or'
          ' ABSENCE.'
      )
  )
  secondary_eim_facets: list[str] = pydantic.Field(
      description='Other EIM facets visually present during this event.'
  )
  clinical_event_description: str = pydantic.Field(
      description=(
          'Concrete physical observation. Must describe what is VISIBLE, not'
          ' inferred. E.g., "Patient clutches right knee during flexion."'
      )
  )
  is_pertinent_negative: bool = pydantic.Field(
      description=(
          'True if this event is a clinical maneuver where an expected sign'
          ' is absent (e.g., no pronator drift, no tremor).'
      )
  )
  transcript_anchor: str = pydantic.Field(
      description=(
          'The utterance (if any) that triggered or accompanied this event.'
          ' Empty string if no verbal anchor.'
      )
  )
  spatial_target: str = pydantic.Field(
      description=(
          'Anatomical target, e.g., "right_upper_quadrant", "left_knee",'
          ' "bilateral_hands", "mid_chest". Use standardized terms.'
      )
  )
  appearance_state: str = pydantic.Field(
      description=(
          'Facial/postural state, e.g., "grimacing", "neutral",'
          ' "eyes_closed", "hunched_forward". Use standardized terms.'
      )
  )
  action_verb: str = pydantic.Field(
      description=(
          'Primary action, e.g., "guarding", "rubbing", "pointing",'
          ' "flexing", "extending", "palpating". Use standardized terms.'
      )
  )


@pydantic.dataclasses.dataclass
class FineScanResult(constraints.BaseModel):
  """Pass 2 output: precise events within one candidate window."""

  events: list[EIMEventCandidate] = pydantic.Field(
      description=(
          'All distinct clinical events visible in the segment. Empty if'
          ' there are none.'
      )
  )


@pydantic.dataclasses.dataclass
class DiscoveredEIMEvent(constraints.BaseModel):
  """A clinically relevant moment discovered by the single-pass scan."""

  start_sec: float = pydantic.Field(
      description=(
          'Start timestamp in seconds where the clinical action/finding begins.'
      )
  )
  end_sec: float = pydantic.Field(
      description=(
          'End timestamp in seconds where the clinical action/finding ends.'
      )
  )
  primary_eim_facet: str = pydantic.Field(
      description=(
          'Primary EIM facet: ACTION, SPATIAL, TEMPORAL, APPEARANCE, or'
          ' ABSENCE.'
      )
  )
  clinical_event_description: str = pydantic.Field(
      description=(
          'Concise, precise medical description of the visual event/maneuver.'
          ' Max 30 words.'
      )
  )
  maneuver_or_finding_name: str = pydantic.Field(
      description=(
          'Standard clinical name of the exam maneuver, procedure step, or'
          ' finding.'
      )
  )
  anatomical_target: str = pydantic.Field(
      description=(
          'Anatomical region or body part involved (e.g. "cervical spine",'
          ' "right knee", "bilateral hands").'
      )
  )
  speech_summary: str = pydantic.Field(
      description=(
          'Verbatim transcription or summary of dialogue/speech during this'
          ' moment.'
      )
  )


@pydantic.dataclasses.dataclass
class VideoDiscoveryResult(constraints.BaseModel):
  """All distinct clinical moments discovered in a full medical video."""

  events: list[DiscoveredEIMEvent] = pydantic.Field(
      description=(
          'List of all non-overlapping, high-yield clinical EIM moments in the'
          ' video.'
      )
  )


# ---------------------------------------------------------------------------
# Configuration and inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DiscoveryConfig:
  """Stage 0 settings; see the flag definitions for their meaning."""

  discovery_mode: str = COARSE_TO_FINE
  transcribe_audio: bool = True
  coarse_fps: float = 5.0
  fine_fps: float = 10.0
  single_pass_fps: float = 5.0
  merge_gap_sec: float = 10.0
  window_padding_sec: float = 5.0
  transcript_padding_sec: float = 5.0
  dedup_scope: str = 'video'
  dedup_max_gap_sec: float = 120.0
  padding_sec: float = 5.0
  min_clip_duration_sec: float = 5.0
  max_clip_duration_sec: float = 60.0


def make_config(
    discovery_mode: str = COARSE_TO_FINE,
    padding_sec: float | None = None,
    max_clip_duration_sec: float | None = None,
    **kwargs: Any,
) -> DiscoveryConfig:
  """Builds and validates a DiscoveryConfig.

  Args:
    discovery_mode: COARSE_TO_FINE or SINGLE_PASS.
    padding_sec: Clip padding. If None, the default of the mode is used.
    max_clip_duration_sec: Maximum clip duration (0 = no maximum). If None, the
      default of the mode is used.
    **kwargs: Other DiscoveryConfig fields.

  Returns:
    The validated config.

  Raises:
    ValueError: If a setting is out of range.
  """
  if discovery_mode not in _MODE_DEFAULTS:
    raise ValueError(f'Unknown discovery mode: {discovery_mode!r}')
  defaults = _MODE_DEFAULTS[discovery_mode]
  config = DiscoveryConfig(
      discovery_mode=discovery_mode,
      padding_sec=(
          defaults['padding_sec'] if padding_sec is None else padding_sec
      ),
      max_clip_duration_sec=(
          defaults['max_clip_duration_sec']
          if max_clip_duration_sec is None
          else max_clip_duration_sec
      ),
      **kwargs,
  )
  for name in ('coarse_fps', 'fine_fps', 'single_pass_fps'):
    fps = getattr(config, name)
    if not 0 < fps <= _MAX_GEMINI_FPS:
      raise ValueError(f'{name} must be in (0, {_MAX_GEMINI_FPS}], got {fps}.')
  for name in (
      'merge_gap_sec',
      'window_padding_sec',
      'transcript_padding_sec',
      'dedup_max_gap_sec',
      'padding_sec',
      'min_clip_duration_sec',
      'max_clip_duration_sec',
  ):
    if getattr(config, name) < 0:
      raise ValueError(f'{name} must be >= 0, got {getattr(config, name)}.')
  if config.dedup_scope not in ('video', 'window'):
    raise ValueError(f'Unknown dedup scope: {config.dedup_scope!r}')
  return config


@dataclasses.dataclass(frozen=True)
class VideoSource:
  """A source video and the ID used to name its clips and transcript."""

  video_id: str
  path: str


def list_videos(
    videos_dir: str, videos_list_file: str, max_videos: int = 0
) -> list[VideoSource]:
  """Lists the source videos to process.

  Args:
    videos_dir: Directory containing .mp4 videos (used if no list file).
    videos_list_file: Text file with one video path per line.
    max_videos: Maximum number of videos to return (0 = all).

  Returns:
    The videos, with IDs taken from the file names without extension.

  Raises:
    ValueError: If no input is given or two videos share an ID.
  """
  if videos_list_file:
    with open(videos_list_file, 'r') as f:
      paths = [line.strip() for line in f if line.strip()]
  elif videos_dir:
    paths = sorted(
        os.path.join(videos_dir, name)
        for name in os.listdir(videos_dir)
        if name.lower().endswith('.mp4')
    )
  else:
    raise ValueError('Set --videos_dir or --videos_list_file.')
  if max_videos > 0:
    paths = paths[:max_videos]

  videos = []
  seen_ids = set()
  for path in paths:
    video_id = os.path.splitext(os.path.basename(path))[0]
    if video_id in seen_ids:
      raise ValueError(
          f'Two videos share the ID {video_id!r} (e.g. {path}); their clip'
          ' names would collide.'
      )
    seen_ids.add(video_id)
    videos.append(VideoSource(video_id=video_id, path=path))
  return videos


def _clean_segments(
    segments: Iterable[TranscriptSegment],
) -> list[TranscriptSegment]:
  """Drops empty or inverted segments and sorts the rest by time."""
  cleaned = [
      s
      for s in segments
      if s.text and s.text.strip() and s.end_sec >= s.start_sec >= 0
  ]
  return sorted(cleaned, key=lambda s: (s.start_sec, s.end_sec))


def load_transcript(path: str) -> list[TranscriptSegment]:
  """Loads a transcript JSON file.

  Args:
    path: JSON file holding a list of {"start_sec", "end_sec", "speaker"
      (optional), "text"} objects (or an object with such a "segments" list).

  Returns:
    The valid segments, sorted by start time. Malformed entries are skipped.
  """
  with open(path, 'r') as f:
    data = json.load(f)
  if isinstance(data, dict):
    data = data.get('segments', [])
  segments = []
  for raw in data:
    try:
      segments.append(
          TranscriptSegment(
              start_sec=float(raw['start_sec']),
              end_sec=float(raw['end_sec']),
              text=str(raw.get('text') or '').strip(),
              speaker=str(raw.get('speaker') or '').strip(),
          )
      )
    except (AttributeError, KeyError, TypeError, ValueError) as e:
      logging.warning('Skipping malformed transcript entry in %s: %s', path, e)
  return _clean_segments(segments)


def format_transcript(segments: Sequence[TranscriptSegment]) -> str:
  """Formats segments as '[start-end] Speaker: text' lines."""
  lines = []
  for s in segments:
    speaker = f'{s.speaker}: ' if s.speaker else ''
    lines.append(f'[{s.start_sec:.1f}-{s.end_sec:.1f}] {speaker}{s.text}')
  return '\n'.join(lines)


def transcript_excerpt(
    segments: Sequence[TranscriptSegment], start_sec: float, end_sec: float
) -> str:
  """Returns the formatted segments that overlap [start_sec, end_sec]."""
  return format_transcript(
      [s for s in segments if s.start_sec < end_sec and s.end_sec > start_sec]
  )


# ---------------------------------------------------------------------------
# Media utilities
# ---------------------------------------------------------------------------


class MediaError(RuntimeError):
  """An ffmpeg or ffprobe call failed or produced an unusable output."""


def _run_ffmpeg(args: Sequence[str]) -> None:
  """Runs ffmpeg with the given arguments (the last one is the output)."""
  cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', *args]
  try:
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
  except subprocess.CalledProcessError as e:
    stderr = (e.stderr or b'').decode(errors='replace').strip()
    raise MediaError(f'ffmpeg failed for {args[-1]}: {stderr[-500:]}') from e
  except OSError as e:
    raise MediaError(f'Could not run ffmpeg: {e}') from e


def _check_output(path: str) -> None:
  """Raises MediaError unless path exists and is not trivially small."""
  if not os.path.exists(path) or os.path.getsize(path) < _MIN_OUTPUT_BYTES:
    raise MediaError(f'ffmpeg output is missing or too small: {path}')


def get_media_duration_sec(path: str) -> float:
  """Returns the duration of a media file in seconds, using ffprobe."""
  cmd = [
      'ffprobe',
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      path,
  ]
  try:
    output = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout
    duration = float(output.strip())
  except (OSError, subprocess.CalledProcessError, ValueError) as e:
    raise MediaError(f'Could not read the duration of {path}: {e}') from e
  if duration <= 0:
    raise MediaError(f'Invalid duration {duration} for {path}.')
  return duration


def has_audio_stream(path: str) -> bool:
  """Returns whether a media file has an audio stream, using ffprobe."""
  cmd = [
      'ffprobe',
      '-v',
      'error',
      '-select_streams',
      'a',
      '-show_entries',
      'stream=index',
      '-of',
      'csv=p=0',
      path,
  ]
  try:
    output = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout
  except (OSError, subprocess.CalledProcessError) as e:
    raise MediaError(f'Could not read the streams of {path}: {e}') from e
  return bool(output.strip())


def cut_video(
    src: str, start_sec: float, end_sec: float, out_path: str, keep_audio: bool
) -> None:
  """Cuts [start_sec, end_sec] from a video, re-encoding for exact boundaries.

  Args:
    src: Source video path.
    start_sec: Start of the cut in seconds.
    end_sec: End of the cut in seconds.
    out_path: Output .mp4 path.
    keep_audio: If False, the output has no audio track.

  Raises:
    MediaError: If ffmpeg fails or the output is missing.
  """
  args = [
      '-ss',
      f'{start_sec:.3f}',
      '-i',
      src,
      '-t',
      f'{end_sec - start_sec:.3f}',
      '-map',
      '0:v:0',
  ]
  if keep_audio:
    args += ['-map', '0:a:0?', '-c:a', 'aac', '-b:a', '128k']
  else:
    args += ['-an']
  args += [
      '-c:v',
      'libx264',
      '-preset',
      'fast',
      '-crf',
      '22',
      '-pix_fmt',
      'yuv420p',
      '-movflags',
      '+faststart',
      out_path,
  ]
  _run_ffmpeg(args)
  _check_output(out_path)


def strip_audio(src: str, out_path: str) -> None:
  """Writes a copy of a video without its audio track."""
  _run_ffmpeg(['-i', src, '-map', '0:v:0', '-c:v', 'copy', '-an', out_path])
  _check_output(out_path)


def extract_audio_wav(src: str, out_path: str) -> None:
  """Extracts the audio track of a video as 16 kHz mono 16-bit PCM WAV."""
  _run_ffmpeg([
      '-i',
      src,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-acodec',
      'pcm_s16le',
      out_path,
  ])
  _check_output(out_path)


def _remove_quietly(*paths: str) -> None:
  """Deletes files, ignoring the ones that do not exist."""
  for path in paths:
    with contextlib.suppress(FileNotFoundError):
      os.remove(path)


@contextlib.contextmanager
def _media_part(
    lm: Any, path: str, mimetype: str, fps: float | None = None
) -> Iterator[dict[str, Any]]:
  """Yields a Gemini part for a local media file.

  Small files are sent inline. Larger files (e.g. full-length recordings) are
  uploaded with the Files API through the language model's client and deleted
  again afterwards (see content_utils.media_part).

  Args:
    lm: The language model; its `client` attribute is used for uploads.
    path: Local media file.
    mimetype: MIME type of the file.
    fps: Video sampling rate, or None for audio / the Gemini default.

  Yields:
    A part dict for the request contents.
  """
  with content_utils.media_part(
      getattr(lm, 'client', None), path, mimetype=mimetype, fps=fps
  ) as part:
    yield part


# ---------------------------------------------------------------------------
# Coarse scans, window fusion and fine localization
# ---------------------------------------------------------------------------


def _normalize_facets(facets: Iterable[str] | None) -> list[str]:
  """Upper-cases facet labels, dropping empty and repeated ones."""
  normalized = []
  for facet in facets or []:
    label = str(facet).strip().upper()
    if label and label not in normalized:
      normalized.append(label)
  return normalized


def _clamp_window(
    start_sec: float, end_sec: float, video_duration: float
) -> tuple[float, float] | None:
  """Orders and clamps a window to the video; None if it lies outside it."""
  start_sec, end_sec = sorted((float(start_sec), float(end_sec)))
  start_sec = max(0.0, start_sec)
  end_sec = min(video_duration, end_sec)
  if end_sec < start_sec:
    return None
  return start_sec, end_sec


def _window_dicts(
    windows: Iterable[LanguageCueWindow | VisualCueWindow],
    video_duration: float,
) -> list[dict[str, Any]]:
  """Converts coarse-scan windows to dicts clamped to the video."""
  result = []
  for w in windows:
    clamped = _clamp_window(w.start_sec, w.end_sec, video_duration)
    if clamped is None:
      continue
    entry = {
        'start_sec': clamped[0],
        'end_sec': clamped[1],
        'description': w.description,
        'suggested_eim_facets': _normalize_facets(w.suggested_eim_facets),
    }
    if isinstance(w, LanguageCueWindow):
      entry['trigger_quote'] = w.trigger_quote
    result.append(entry)
  return result


def transcribe_audio(
    video_path: str, lm: language_model.LanguageModel, work_dir: str
) -> list[TranscriptSegment]:
  """Transcribes the audio track of a video with an audio-only Gemini call."""
  wav_path = os.path.join(work_dir, 'full_audio.wav')
  extract_audio_wav(video_path, wav_path)
  prompt = template_utils.render_template('stage0_transcribe_audio_prompt')
  prompt += constraints.get_constraint_suffix(cls=TranscriptionResult)
  with _media_part(lm, wav_path, _AUDIO_MIMETYPE) as audio_part:
    result, _ = lm.generate_thought_and_object(
        [audio_part, prompt], cls=TranscriptionResult
    )
  return _clean_segments(result.segments)


def run_language_scan(
    transcript_text: str,
    lm: language_model.LanguageModel,
    video_duration: float,
) -> list[dict[str, Any]]:
  """Pass 1A: proposes candidate windows from the timestamped transcript."""
  prompt = template_utils.render_template(
      'stage0_language_scan_prompt', transcript=transcript_text
  )
  prompt += constraints.get_constraint_suffix(cls=LanguageScanResult)
  result, _ = lm.generate_thought_and_object([prompt], cls=LanguageScanResult)
  return _window_dicts(result.candidate_windows, video_duration)


def run_visual_scan(
    video_path: str,
    lm: language_model.LanguageModel,
    video_duration: float,
    fps: float,
    work_dir: str,
) -> list[dict[str, Any]]:
  """Pass 1B: proposes candidate windows from the muted full video."""
  muted_path = os.path.join(work_dir, 'full_noaudio.mp4')
  strip_audio(video_path, muted_path)
  prompt = template_utils.render_template('stage0_visual_scan_prompt')
  prompt += constraints.get_constraint_suffix(cls=VisualScanResult)
  try:
    with _media_part(lm, muted_path, _VIDEO_MIMETYPE, fps=fps) as video_part:
      result, _ = lm.generate_thought_and_object(
          [video_part, prompt], cls=VisualScanResult
      )
  finally:
    _remove_quietly(muted_path)
  return _window_dicts(result.candidate_windows, video_duration)


def merge_candidate_windows(
    language_windows: Sequence[dict[str, Any]],
    visual_windows: Sequence[dict[str, Any]],
    merge_gap_sec: float = 10.0,
) -> list[dict[str, Any]]:
  """Fuses the language and visual candidate windows.

  All windows are sorted by start time. A window is merged into the previous
  (merged) window iff it starts at most merge_gap_sec after that window ends
  and the two share at least one EIM facet. Facet labels are compared
  case-insensitively.

  Args:
    language_windows: Pass 1A windows.
    visual_windows: Pass 1B windows.
    merge_gap_sec: Maximum gap between merged windows.

  Returns:
    Merged windows sorted by start time, each with 'start_sec', 'end_sec',
    'description', 'eim_facets', 'sources' (TRANSCRIPT and/or VISUAL) and
    'trigger_quotes'.
  """
  windows = []
  for source, source_windows in (
      (TRANSCRIPT_SOURCE, language_windows),
      (VISUAL_SOURCE, visual_windows),
  ):
    for w in source_windows:
      windows.append({
          'start_sec': float(w['start_sec']),
          'end_sec': float(w['end_sec']),
          'description': w.get('description', ''),
          'eim_facets': set(_normalize_facets(w.get('suggested_eim_facets'))),
          'sources': {source},
          'trigger_quotes': (
              [w['trigger_quote']] if w.get('trigger_quote') else []
          ),
      })
  windows.sort(key=lambda w: w['start_sec'])

  merged = []
  for w in windows:
    prev = merged[-1] if merged else None
    if (
        prev is not None
        and w['start_sec'] <= prev['end_sec'] + merge_gap_sec
        and w['eim_facets'] & prev['eim_facets']
    ):
      prev['end_sec'] = max(prev['end_sec'], w['end_sec'])
      prev['description'] += f' | {w["description"]}'
      prev['eim_facets'] |= w['eim_facets']
      prev['sources'] |= w['sources']
      prev['trigger_quotes'] += w['trigger_quotes']
    else:
      merged.append(w)

  for w in merged:
    w['eim_facets'] = sorted(w['eim_facets'])
    w['sources'] = sorted(w['sources'])
  return merged


def to_absolute_event(
    candidate: EIMEventCandidate, segment_start: float, segment_end: float
) -> dict[str, Any] | None:
  """Converts a fine-scan event to absolute video time.

  Args:
    candidate: Event with timestamps relative to the segment start.
    segment_start: Start of the scanned segment in the full video.
    segment_end: End of the scanned segment in the full video.

  Returns:
    The event dict with absolute timestamps (rounded to 0.1 s) clamped to the
    segment, or None if the event ends at or before its start.
  """
  low, high = segment_start, segment_end
  start_sec = round(min(max(low + candidate.start_sec, low), high), 1)
  end_sec = round(min(max(low + candidate.end_sec, low), high), 1)
  if end_sec <= start_sec:
    return None
  return {
      'start_sec': start_sec,
      'end_sec': end_sec,
      'primary_eim_facet': candidate.primary_eim_facet.strip().upper(),
      'secondary_eim_facets': _normalize_facets(candidate.secondary_eim_facets),
      'clinical_event_description': (
          candidate.clinical_event_description.strip()
      ),
      'is_pertinent_negative': bool(candidate.is_pertinent_negative),
      'transcript_anchor': candidate.transcript_anchor.strip(),
      'spatial_target': candidate.spatial_target.strip(),
      'appearance_state': candidate.appearance_state.strip(),
      'action_verb': candidate.action_verb.strip(),
  }


def run_fine_scan(
    video_path: str,
    window: dict[str, Any],
    window_index: int,
    video_duration: float,
    segments: Sequence[TranscriptSegment],
    lm: language_model.LanguageModel,
    config: DiscoveryConfig,
    work_dir: str,
) -> list[dict[str, Any]]:
  """Pass 2: localizes events in one merged window at the fine frame rate.

  Args:
    video_path: Source video.
    window: Merged candidate window.
    window_index: Index of the window (for file names and provenance).
    video_duration: Duration of the source video in seconds.
    segments: Transcript of the video (may be empty).
    lm: Language model.
    config: Stage 0 settings.
    work_dir: Directory for temporary files.

  Returns:
    Valid events with absolute timestamps.

  Raises:
    MediaError: If the segment cannot be cut.
    Exception: Model or parsing errors are propagated to the caller.
  """
  segment_start = max(0.0, window['start_sec'] - config.window_padding_sec)
  segment_end = min(
      video_duration, window['end_sec'] + config.window_padding_sec
  )
  if segment_end - segment_start < _MIN_SEGMENT_SEC:
    raise MediaError(
        f'Window {segment_start:.1f}-{segment_end:.1f}s is too short to scan.'
    )
  clip_path = os.path.join(work_dir, f'window_{window_index:03d}_noaudio.mp4')
  excerpt = transcript_excerpt(
      segments,
      segment_start - config.transcript_padding_sec,
      segment_end + config.transcript_padding_sec,
  )
  prompt = template_utils.render_template(
      'stage0_fine_scan_prompt',
      segment_start=f'{segment_start:.1f}',
      segment_end=f'{segment_end:.1f}',
      transcript=excerpt or _NO_TRANSCRIPT,
  )
  prompt += constraints.get_constraint_suffix(cls=FineScanResult)
  try:
    cut_video(video_path, segment_start, segment_end, clip_path, False)
    with _media_part(
        lm, clip_path, _VIDEO_MIMETYPE, fps=config.fine_fps
    ) as video_part:
      result, _ = lm.generate_thought_and_object(
          [video_part, prompt], cls=FineScanResult
      )
  finally:
    _remove_quietly(clip_path)

  events = []
  for candidate in result.events:
    event = to_absolute_event(candidate, segment_start, segment_end)
    if event is None:
      logging.info(
          '  Dropping event with invalid timestamps in window %d.',
          window_index,
      )
      continue
    event['window_index'] = window_index
    event['window_sources'] = list(window['sources'])
    events.append(event)
  return events


def _normalize_token(value: Any) -> str:
  """Normalizes a structured field so 'Right_Knee' matches 'right knee'."""
  return re.sub(r'[\s_\-]+', ' ', str(value or '')).strip().lower()


_DEDUP_FIELDS = (
    'primary_eim_facet',
    'spatial_target',
    'appearance_state',
    'action_verb',
)


def is_duplicate(
    later: dict[str, Any],
    earlier: dict[str, Any],
    scope: str = 'video',
    max_gap_sec: float = 120.0,
) -> bool:
  """Returns True if `later` repeats the persistent sign in `earlier`.

  Uses structured fields rather than text similarity, since deduplication is
  a clinical judgment about whether the underlying physical state changed.

  Args:
    later: The event that starts later.
    earlier: An event kept earlier.
    scope: 'video' (repeats anywhere in the video) or 'window' (repeats that
      start at most max_gap_sec after `earlier` ends).
    max_gap_sec: Maximum gap for the 'window' scope.

  Returns:
    Whether `later` is a duplicate of `earlier`.
  """
  for field in _DEDUP_FIELDS:
    if _normalize_token(later.get(field)) != _normalize_token(
        earlier.get(field)
    ):
      return False
  if bool(later.get('is_pertinent_negative')) != bool(
      earlier.get('is_pertinent_negative')
  ):
    return False
  if scope == 'video':
    return True
  if scope == 'window':
    return later['start_sec'] - earlier['end_sec'] <= max_gap_sec
  raise ValueError(f'Unknown dedup scope: {scope!r}')


def deduplicate_events(
    events: Sequence[dict[str, Any]],
    scope: str = 'video',
    max_gap_sec: float = 120.0,
) -> list[dict[str, Any]]:
  """Removes repeated events, keeping the first occurrence."""
  kept = []
  for event in sorted(events, key=lambda e: (e['start_sec'], e['end_sec'])):
    if not any(
        is_duplicate(event, k, scope=scope, max_gap_sec=max_gap_sec)
        for k in kept
    ):
      kept.append(event)
  return kept


def plan_clip_span(
    event_start: float,
    event_end: float,
    video_duration: float,
    padding_sec: float,
    min_duration_sec: float,
    max_duration_sec: float,
) -> tuple[float, float, bool] | None:
  """Pads an event into a clip span.

  Args:
    event_start: Event start in seconds.
    event_end: Event end in seconds.
    video_duration: Duration of the source video, used for clamping.
    padding_sec: Padding added to each side of the event.
    min_duration_sec: Spans shorter than this are rejected.
    max_duration_sec: Longer spans are truncated to this length (0 = no
      maximum).

  Returns:
    (start, end, truncated), or None if the padded span is too short.
  """
  start = max(0.0, event_start - padding_sec)
  end = min(video_duration, event_end + padding_sec)
  if end - start < min_duration_sec:
    return None
  truncated = 0 < max_duration_sec < end - start
  if truncated:
    end = start + max_duration_sec
  return start, end, truncated


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------


def _new_video_record(video: VideoSource) -> dict[str, Any]:
  return {
      'video_id': video.video_id,
      'video_path': video.path,
      'status': 'error',
      'duration_sec': None,
      'transcript_source': 'none',
      'counts': {
          'transcript_segments': 0,
          'pass1a_windows': 0,
          'pass1b_windows': 0,
          'merged_windows': 0,
          'window_errors': 0,
          'pre_dedup_events': 0,
          'post_dedup_events': 0,
          'skipped_short_events': 0,
          'truncated_events': 0,
          'clip_errors': 0,
          'clips': 0,
      },
      'errors': [],
  }


def _record_error(record: dict[str, Any], step: str, error: Exception) -> None:
  message = f'{step}: {type(error).__name__}: {error}'
  logging.warning('[%s] %s', record['video_id'], message)
  record['errors'].append(message[:1000])


def _get_transcript(
    video: VideoSource,
    lm: language_model.LanguageModel,
    config: DiscoveryConfig,
    transcripts_dir: str,
    work_dir: str,
    record: dict[str, Any],
) -> list[TranscriptSegment]:
  """Loads the provided transcript or transcribes the audio track."""
  if transcripts_dir:
    path = os.path.join(transcripts_dir, f'{video.video_id}.json')
    if os.path.exists(path):
      record['transcript_source'] = 'provided'
      return load_transcript(path)
  if not config.transcribe_audio:
    return []
  try:
    segments = transcribe_audio(video.path, lm, work_dir)
  except Exception as e:  # pylint: disable=broad-except
    _record_error(record, 'transcription', e)
    return []
  record['transcript_source'] = 'gemini_transcription'
  return segments


def _discover_coarse_to_fine(
    video: VideoSource,
    video_duration: float,
    lm: language_model.LanguageModel,
    config: DiscoveryConfig,
    transcripts_dir: str,
    work_dir: str,
    record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[TranscriptSegment]]:
  """Runs the coarse-to-fine scan; returns the deduplicated events."""
  counts = record['counts']
  segments = _get_transcript(
      video, lm, config, transcripts_dir, work_dir, record
  )
  counts['transcript_segments'] = len(segments)

  coarse_scan_succeeded = False
  language_windows = []
  if segments:
    try:
      language_windows = run_language_scan(
          format_transcript(segments), lm, video_duration
      )
      coarse_scan_succeeded = True
    except Exception as e:  # pylint: disable=broad-except
      _record_error(record, 'pass1a_language_scan', e)
  visual_windows = []
  try:
    visual_windows = run_visual_scan(
        video.path, lm, video_duration, config.coarse_fps, work_dir
    )
    coarse_scan_succeeded = True
  except Exception as e:  # pylint: disable=broad-except
    _record_error(record, 'pass1b_visual_scan', e)
  if not coarse_scan_succeeded:
    raise RuntimeError('No coarse scan succeeded.')
  counts['pass1a_windows'] = len(language_windows)
  counts['pass1b_windows'] = len(visual_windows)

  merged = merge_candidate_windows(
      language_windows, visual_windows, config.merge_gap_sec
  )
  counts['merged_windows'] = len(merged)
  record['windows'] = [
      {
          'start_sec': round(w['start_sec'], 2),
          'end_sec': round(w['end_sec'], 2),
          'eim_facets': w['eim_facets'],
          'sources': w['sources'],
      }
      for w in merged
  ]
  logging.info(
      '[%s] %d language + %d visual windows -> %d merged windows.',
      video.video_id,
      len(language_windows),
      len(visual_windows),
      len(merged),
  )

  events = []
  for window_index, window in enumerate(merged):
    try:
      events.extend(
          run_fine_scan(
              video.path,
              window,
              window_index,
              video_duration,
              segments,
              lm,
              config,
              work_dir,
          )
      )
    except Exception as e:  # pylint: disable=broad-except
      counts['window_errors'] += 1
      _record_error(record, f'pass2_window_{window_index}', e)
  if merged and counts['window_errors'] == len(merged):
    raise RuntimeError(f'All {len(merged)} fine-scan windows failed.')

  counts['pre_dedup_events'] = len(events)
  deduped = deduplicate_events(
      events, scope=config.dedup_scope, max_gap_sec=config.dedup_max_gap_sec
  )
  counts['post_dedup_events'] = len(deduped)
  return deduped, segments


def _discover_single_pass(
    video: VideoSource,
    lm: language_model.LanguageModel,
    config: DiscoveryConfig,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
  """Runs one discovery call on the full video with audio."""
  record['transcript_source'] = 'speech_summary'
  prompt = template_utils.render_template('stage0_single_pass_prompt')
  prompt += constraints.get_constraint_suffix(cls=VideoDiscoveryResult)
  with _media_part(
      lm, video.path, _VIDEO_MIMETYPE, fps=config.single_pass_fps
  ) as video_part:
    result, _ = lm.generate_thought_and_object(
        [video_part, prompt], cls=VideoDiscoveryResult
    )
  events = [
      {
          'start_sec': float(e.start_sec),
          'end_sec': float(e.end_sec),
          'primary_eim_facet': e.primary_eim_facet.strip().upper(),
          'clinical_event_description': e.clinical_event_description.strip(),
          'maneuver_name': e.maneuver_or_finding_name,
          'anatomical_target': e.anatomical_target,
          'speech_summary': e.speech_summary,
      }
      for e in result.events
  ]
  record['counts']['pre_dedup_events'] = len(events)
  record['counts']['post_dedup_events'] = len(events)
  return events


_EVENT_METADATA_KEYS = (
    'secondary_eim_facets',
    'is_pertinent_negative',
    'transcript_anchor',
    'spatial_target',
    'appearance_state',
    'action_verb',
    'window_sources',
    'maneuver_name',
    'anatomical_target',
)


def extract_clips(
    events: Sequence[dict[str, Any]],
    video: VideoSource,
    video_duration: float,
    segments: Sequence[TranscriptSegment],
    output_clips_dir: str,
    config: DiscoveryConfig,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
  """Cuts the final clips for the events and returns their manifest entries.

  Each clip is written as <clip>.mp4 (video and audio), <clip>_noaudio.mp4
  and <clip>_audio.wav (16 kHz mono). If the source video has no audio track,
  the WAV file is not written and the entry's 'clip_path_audio' is ''. A clip
  is dropped if any of its outputs fails.

  Args:
    events: Events with absolute 'start_sec' / 'end_sec'.
    video: Source video.
    video_duration: Duration of the source video in seconds.
    segments: Transcript, used for the speech within each clip.
    output_clips_dir: Output directory.
    config: Stage 0 settings.
    record: Per-video record; counts and errors are updated in place.

  Returns:
    One manifest entry per extracted clip.
  """
  counts = record['counts']
  clips = []
  for event in events:
    span = plan_clip_span(
        event['start_sec'],
        event['end_sec'],
        video_duration,
        config.padding_sec,
        config.min_clip_duration_sec,
        config.max_clip_duration_sec,
    )
    if span is None:
      counts['skipped_short_events'] += 1
      continue
    start, end, truncated = span
    if truncated:
      counts['truncated_events'] += 1

    clip_id = f'{video.video_id}_stage0_clip_{len(clips) + 1:02d}'
    clip_path = os.path.join(output_clips_dir, f'{clip_id}.mp4')
    noaudio_path = os.path.join(output_clips_dir, f'{clip_id}_noaudio.mp4')
    audio_path = os.path.join(output_clips_dir, f'{clip_id}_audio.wav')
    try:
      cut_video(video.path, start, end, clip_path, keep_audio=True)
      strip_audio(clip_path, noaudio_path)
      clip_has_audio = has_audio_stream(clip_path)
      if clip_has_audio:
        extract_audio_wav(clip_path, audio_path)
    except MediaError as e:
      counts['clip_errors'] += 1
      _record_error(record, clip_id, e)
      _remove_quietly(clip_path, noaudio_path, audio_path)
      continue

    if config.discovery_mode == SINGLE_PASS:
      transcript = event.get('speech_summary') or _NO_SPEECH
    else:
      transcript = transcript_excerpt(segments, start, end)
    entry = {
        'clip_id': clip_id,
        'clip_file': f'{clip_id}.mp4',
        'clip_path': os.path.abspath(clip_path),
        'clip_path_noaudio': os.path.abspath(noaudio_path),
        'clip_path_audio': (
            os.path.abspath(audio_path) if clip_has_audio else ''
        ),
        'source_video_id': video.video_id,
        'source_video_path': video.path,
        'discovery_mode': config.discovery_mode,
        'start_sec': round(start, 2),
        'end_sec': round(end, 2),
        'clip_duration_sec': round(end - start, 2),
        'event_start_sec': event['start_sec'],
        'event_end_sec': event['end_sec'],
        'padding_sec': config.padding_sec,
        'primary_eim_facet': event['primary_eim_facet'],
        'clinical_event_description': event['clinical_event_description'],
        'transcript': transcript,
    }
    entry.update({k: event[k] for k in _EVENT_METADATA_KEYS if k in event})
    clips.append(entry)
  counts['clips'] = len(clips)
  return clips


def process_video(
    video: VideoSource,
    lm: language_model.LanguageModel,
    config: DiscoveryConfig,
    transcripts_dir: str,
    output_clips_dir: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Discovers events in one video and extracts their clips.

  Errors are recorded in the returned per-video record rather than raised:
  a failed coarse scan or fine-scan window only removes its own candidates,
  and the video is marked 'error' only if nothing could be scanned.

  Args:
    video: Source video.
    lm: Language model.
    config: Stage 0 settings.
    transcripts_dir: Optional directory of transcripts.
    output_clips_dir: Output directory for clips.

  Returns:
    (per-video record, clip manifest entries). The record's status is
    'success', 'no_events' or 'error'.
  """
  record = _new_video_record(video)
  logging.info('Processing %s (%s mode)...', video.path, config.discovery_mode)
  try:
    video_duration = get_media_duration_sec(video.path)
    record['duration_sec'] = round(video_duration, 2)
    with tempfile.TemporaryDirectory(
        prefix=f'stage0_{video.video_id}_'
    ) as work_dir:
      if config.discovery_mode == SINGLE_PASS:
        events = _discover_single_pass(video, lm, config, record)
        segments = []
      else:
        events, segments = _discover_coarse_to_fine(
            video,
            video_duration,
            lm,
            config,
            transcripts_dir,
            work_dir,
            record,
        )
      clips = extract_clips(
          events,
          video,
          video_duration,
          segments,
          output_clips_dir,
          config,
          record,
      )
  except Exception as e:  # pylint: disable=broad-except
    logging.exception('Stage 0 failed for %s', video.path)
    record['status'] = 'error'
    record['error'] = f'{type(e).__name__}: {e}'
    return record, []

  record['status'] = 'success' if clips else 'no_events'
  logging.info(
      '[%s] %s: %d clips (%d events before dedup).',
      video.video_id,
      record['status'],
      len(clips),
      record['counts']['pre_dedup_events'],
  )
  return record, clips


# ---------------------------------------------------------------------------
# Manifest and orchestration
# ---------------------------------------------------------------------------


def build_manifest(
    video_records: Sequence[dict[str, Any]],
    clips: Sequence[dict[str, Any]],
    model_name: str,
    config: DiscoveryConfig,
) -> dict[str, Any]:
  """Assembles the Stage 0 manifest consumed by stage1_prototype.py."""
  if config.discovery_mode == SINGLE_PASS:
    fps_settings = {'single_pass_fps': config.single_pass_fps}
  else:
    fps_settings = {
        'coarse_fps': config.coarse_fps,
        'fine_fps': config.fine_fps,
    }
  settings = {
      k: v
      for k, v in dataclasses.asdict(config).items()
      if not k.endswith('_fps') and k != 'discovery_mode'
  }
  return {
      'stage': 'stage0_discovery',
      'discovery_mode': config.discovery_mode,
      'model': model_name,
      **fps_settings,
      'settings': settings,
      'num_videos': len(video_records),
      'num_clips': len(clips),
      'video_status_counts': dict(
          collections.Counter(r['status'] for r in video_records)
      ),
      'videos': sorted(video_records, key=lambda r: r['video_id']),
      'clips': sorted(
          clips, key=lambda c: (c['source_video_id'], c['clip_id'])
      ),
  }


def write_manifest(path: str, manifest: dict[str, Any]) -> None:
  """Writes the manifest atomically."""
  tmp_path = f'{path}.tmp'
  with open(tmp_path, 'w') as f:
    json.dump(manifest, f, indent=2)
  os.replace(tmp_path, path)


def _load_checkpoint(
    path: str, discovery_mode: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Returns the finished videos and their clips from an earlier run."""
  if not os.path.exists(path):
    return [], []
  with open(path, 'r') as f:
    previous = json.load(f)
  if previous.get('discovery_mode') != discovery_mode:
    raise ValueError(
        f'{path} was written with discovery_mode='
        f'{previous.get("discovery_mode")}; use another --output_manifest.'
    )
  records = [
      r
      for r in previous.get('videos', [])
      if r.get('status') in ('success', 'no_events')
  ]
  done_ids = {r['video_id'] for r in records}
  clips = [
      c
      for c in previous.get('clips', [])
      if c.get('source_video_id') in done_ids
  ]
  logging.info('Resuming from %s: %d videos already done.', path, len(records))
  return records, clips


def run_discovery(
    videos: Sequence[VideoSource],
    lm: language_model.LanguageModel,
    config: DiscoveryConfig,
    transcripts_dir: str,
    output_clips_dir: str,
    output_manifest: str,
    model_name: str,
    num_workers: int = 1,
) -> dict[str, Any]:
  """Processes all videos in parallel, checkpointing the manifest.

  Args:
    videos: Source videos.
    lm: Language model.
    config: Stage 0 settings.
    transcripts_dir: Optional directory of transcripts.
    output_clips_dir: Output directory for clips.
    output_manifest: Manifest path. If it exists, videos it lists as done are
      skipped.
    model_name: Model ID recorded in the manifest.
    num_workers: Number of videos processed in parallel.

  Returns:
    The final manifest.
  """
  records, clips = _load_checkpoint(output_manifest, config.discovery_mode)
  done_ids = {r['video_id'] for r in records}
  pending = [v for v in videos if v.video_id not in done_ids]
  logging.info('%d videos to process.', len(pending))
  lock = threading.Lock()

  def _process_and_save(video: VideoSource) -> None:
    record, video_clips = process_video(
        video, lm, config, transcripts_dir, output_clips_dir
    )
    with lock:
      records.append(record)
      clips.extend(video_clips)
      write_manifest(
          output_manifest, build_manifest(records, clips, model_name, config)
      )
      logging.info(
          '[%d/%d] %s: %s',
          len(records),
          len(videos),
          video.video_id,
          record['status'],
      )

  with futures.ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
    list(executor.map(_process_and_save, pending))

  manifest = build_manifest(records, clips, model_name, config)
  write_manifest(output_manifest, manifest)
  return manifest


def main(argv: Sequence[str]) -> None:
  """Runs Stage 0 discovery over the input videos."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  try:
    config = make_config(
        discovery_mode=_DISCOVERY_MODE.value,
        padding_sec=_PADDING_SEC.value,
        max_clip_duration_sec=_MAX_CLIP_DURATION_SEC.value,
        transcribe_audio=_TRANSCRIBE_AUDIO.value,
        coarse_fps=_COARSE_FPS.value,
        fine_fps=_FINE_FPS.value,
        single_pass_fps=_SINGLE_PASS_FPS.value,
        merge_gap_sec=_MERGE_GAP_SEC.value,
        window_padding_sec=_WINDOW_PADDING_SEC.value,
        transcript_padding_sec=_TRANSCRIPT_PADDING_SEC.value,
        dedup_scope=_DEDUP_SCOPE.value,
        dedup_max_gap_sec=_DEDUP_MAX_GAP_SEC.value,
        min_clip_duration_sec=_MIN_CLIP_DURATION_SEC.value,
    )
    videos = list_videos(
        _VIDEOS_DIR.value, _VIDEOS_LIST_FILE.value, _MAX_VIDEOS.value
    )
  except ValueError as e:
    raise app.UsageError(str(e)) from e

  os.makedirs(_OUTPUT_CLIPS_DIR.value, exist_ok=True)
  manifest_dir = os.path.dirname(_OUTPUT_MANIFEST.value)
  if manifest_dir:
    os.makedirs(manifest_dir, exist_ok=True)

  logging.info(
      'Stage 0 (%s): %d videos, padding %.1fs, max clip %.1fs.',
      config.discovery_mode,
      len(videos),
      config.padding_sec,
      config.max_clip_duration_sec,
  )
  lm = language_model.LanguageModel(model_name=_MODEL_NAME.value)
  manifest = run_discovery(
      videos=videos,
      lm=lm,
      config=config,
      transcripts_dir=_TRANSCRIPTS_DIR.value,
      output_clips_dir=_OUTPUT_CLIPS_DIR.value,
      output_manifest=_OUTPUT_MANIFEST.value,
      model_name=_MODEL_NAME.value,
      num_workers=_NUM_WORKERS.value,
  )
  logging.info(
      'Stage 0 complete: %d clips from %d videos %s. Manifest: %s',
      manifest['num_clips'],
      manifest['num_videos'],
      manifest['video_status_counts'],
      _OUTPUT_MANIFEST.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_clips_dir', 'output_manifest'])
  app.run(main)
