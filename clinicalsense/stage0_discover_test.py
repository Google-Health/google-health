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
"""Unit tests for Stage 0 clinical event discovery."""

import json
import os
import subprocess
from unittest import mock

from absl.testing import absltest

import stage0_discover

# Clip manifest fields read by stage1_prototype.py (--stage0_manifest).
_STAGE1_CLIP_FIELDS = (
    'clip_file',
    'clip_path',
    'clip_path_noaudio',
    'clip_path_audio',
    'source_video_id',
    'start_sec',
    'end_sec',
    'event_start_sec',
    'event_end_sec',
    'clip_duration_sec',
    'primary_eim_facet',
    'clinical_event_description',
    'transcript',
)

_LanguageWindow = stage0_discover.LanguageCueWindow
_VisualWindow = stage0_discover.VisualCueWindow


def _event(
    start: float,
    end: float,
    facet: str = 'ACTION',
    spatial: str = 'left_knee',
    action: str = 'flexing',
) -> stage0_discover.EIMEventCandidate:
  return stage0_discover.EIMEventCandidate(
      start_sec=start,
      end_sec=end,
      primary_eim_facet=facet,
      secondary_eim_facets=['spatial'],
      clinical_event_description=f' Patient is {action} ({spatial}). ',
      is_pertinent_negative=False,
      transcript_anchor='Bend your knee.',
      spatial_target=spatial,
      appearance_state='neutral',
      action_verb=action,
  )


def _event_dict(start, end, **overrides):
  event = {
      'start_sec': start,
      'end_sec': end,
      'primary_eim_facet': 'ACTION',
      'spatial_target': 'left_knee',
      'appearance_state': 'neutral',
      'action_verb': 'flexing',
      'is_pertinent_negative': False,
  }
  event.update(overrides)
  return event


class FakeLanguageModel:
  """Returns canned responses, in order, for each requested output schema."""

  def __init__(self, responses):
    self._responses = {cls: list(values) for cls, values in responses.items()}
    self.calls = []

  def generate_thought_and_object(self, prompt, cls, **unused_kwargs):
    self.calls.append((cls, prompt))
    response = self._responses[cls].pop(0)
    if isinstance(response, Exception):
      raise response
    return response, ''

  def contents_for(self, cls):
    return [contents for c, contents in self.calls if c is cls]


class MergeCandidateWindowsTest(absltest.TestCase):

  def test_merges_overlapping_windows_with_shared_facet(self):
    merged = stage0_discover.merge_candidate_windows(
        [{
            'start_sec': 10.0,
            'end_sec': 20.0,
            'description': 'knee flexion',
            'suggested_eim_facets': ['action'],
            'trigger_quote': 'Bend your knee.',
        }],
        [{
            'start_sec': 15.0,
            'end_sec': 30.0,
            'description': 'flexing',
            'suggested_eim_facets': ['ACTION', 'TEMPORAL'],
        }],
    )
    self.assertLen(merged, 1)
    self.assertEqual(merged[0]['start_sec'], 10.0)
    self.assertEqual(merged[0]['end_sec'], 30.0)
    self.assertEqual(merged[0]['eim_facets'], ['ACTION', 'TEMPORAL'])
    self.assertEqual(merged[0]['sources'], ['TRANSCRIPT', 'VISUAL'])
    self.assertEqual(merged[0]['trigger_quotes'], ['Bend your knee.'])
    self.assertEqual(merged[0]['description'], 'knee flexion | flexing')

  def test_keeps_windows_with_disjoint_facets_separate(self):
    merged = stage0_discover.merge_candidate_windows(
        [{'start_sec': 10, 'end_sec': 20, 'suggested_eim_facets': ['ACTION']}],
        [{'start_sec': 12, 'end_sec': 25, 'suggested_eim_facets': ['ABSENCE']}],
    )
    self.assertLen(merged, 2)
    self.assertEqual(
        [w['sources'] for w in merged], [['TRANSCRIPT'], ['VISUAL']]
    )

  def test_merge_gap_boundary(self):
    def _merge(next_start):
      return stage0_discover.merge_candidate_windows(
          [{
              'start_sec': next_start,
              'end_sec': 25,
              'suggested_eim_facets': ['SPATIAL'],
          }],
          [{
              'start_sec': 0,
              'end_sec': 10,
              'suggested_eim_facets': ['SPATIAL'],
          }],
          merge_gap_sec=10.0,
      )

    self.assertLen(_merge(20.0), 1)
    self.assertLen(_merge(20.5), 2)

  def test_output_is_sorted_by_start(self):
    merged = stage0_discover.merge_candidate_windows(
        [{'start_sec': 90, 'end_sec': 95, 'suggested_eim_facets': ['ACTION']}],
        [{'start_sec': 5, 'end_sec': 8, 'suggested_eim_facets': ['ACTION']}],
    )
    self.assertEqual([w['start_sec'] for w in merged], [5.0, 90.0])

  def test_no_windows(self):
    self.assertEmpty(stage0_discover.merge_candidate_windows([], []))


class DeduplicationTest(absltest.TestCase):

  def test_structured_fields_are_normalized(self):
    earlier = _event_dict(0, 5, spatial_target='Right_Knee')
    later = _event_dict(
        100,
        105,
        primary_eim_facet='action',
        spatial_target='right knee',
        appearance_state='Neutral',
        action_verb='Flexing',
    )
    self.assertTrue(stage0_discover.is_duplicate(later, earlier))

  def test_different_fields_are_not_duplicates(self):
    earlier = _event_dict(0, 5)
    self.assertFalse(
        stage0_discover.is_duplicate(
            _event_dict(10, 15, action_verb='extending'), earlier
        )
    )
    self.assertFalse(
        stage0_discover.is_duplicate(
            _event_dict(10, 15, is_pertinent_negative=True), earlier
        )
    )

  def test_window_scope_measures_gap_after_earlier_event(self):
    earlier = _event_dict(0, 10)
    later = _event_dict(100, 110)  # Starts 90 s after `earlier` ends.
    self.assertTrue(
        stage0_discover.is_duplicate(
            later, earlier, scope='window', max_gap_sec=95
        )
    )
    self.assertFalse(
        stage0_discover.is_duplicate(
            later, earlier, scope='window', max_gap_sec=80
        )
    )

  def test_window_scope_overlapping_events_are_duplicates(self):
    self.assertTrue(
        stage0_discover.is_duplicate(
            _event_dict(5, 12),
            _event_dict(0, 10),
            scope='window',
            max_gap_sec=0,
        )
    )

  def test_deduplicate_events_keeps_first_occurrence(self):
    events = [
        _event_dict(50, 55),
        _event_dict(10, 15),
        _event_dict(30, 35, action_verb='pointing'),
    ]
    kept = stage0_discover.deduplicate_events(events)
    self.assertEqual([e['start_sec'] for e in kept], [10, 30])


class ClipSpanTest(absltest.TestCase):

  def test_padding_is_clamped_to_video(self):
    self.assertEqual(
        stage0_discover.plan_clip_span(2.0, 10.0, 12.0, 5.0, 5.0, 60.0),
        (0.0, 12.0, False),
    )

  def test_short_span_is_rejected(self):
    self.assertIsNone(
        stage0_discover.plan_clip_span(10.0, 10.5, 100.0, 2.0, 5.0, 0.0)
    )

  def test_long_span_is_truncated(self):
    self.assertEqual(
        stage0_discover.plan_clip_span(10.0, 100.0, 200.0, 5.0, 5.0, 60.0),
        (5.0, 65.0, True),
    )

  def test_zero_max_means_no_maximum(self):
    self.assertEqual(
        stage0_discover.plan_clip_span(10.0, 100.0, 200.0, 2.0, 5.0, 0.0),
        (8.0, 102.0, False),
    )


class ToAbsoluteEventTest(absltest.TestCase):

  def test_converts_relative_timestamps(self):
    event = stage0_discover.to_absolute_event(
        _event(1.04, 3.46), segment_start=100.0, segment_end=130.0
    )
    self.assertEqual(event['start_sec'], 101.0)
    self.assertEqual(event['end_sec'], 103.5)
    self.assertEqual(event['secondary_eim_facets'], ['SPATIAL'])
    self.assertEqual(
        event['clinical_event_description'], 'Patient is flexing (left_knee).'
    )

  def test_clamps_to_segment(self):
    event = stage0_discover.to_absolute_event(_event(25.0, 40.0), 100.0, 130.0)
    self.assertEqual((event['start_sec'], event['end_sec']), (125.0, 130.0))

  def test_drops_events_that_end_before_they_start(self):
    for start, end in ((5.0, 5.0), (6.0, 2.0), (40.0, 50.0)):
      self.assertIsNone(
          stage0_discover.to_absolute_event(_event(start, end), 100.0, 130.0)
      )


class TranscriptTest(absltest.TestCase):

  def _segments(self):
    return [
        stage0_discover.TranscriptSegment(0.0, 4.0, 'Hello.', 'Doctor'),
        stage0_discover.TranscriptSegment(5.0, 9.0, 'Bend your knee.'),
        stage0_discover.TranscriptSegment(10.0, 14.0, 'Thank you.', 'Doctor'),
    ]

  def test_excerpt_selects_overlapping_segments(self):
    self.assertEqual(
        stage0_discover.transcript_excerpt(self._segments(), 4.0, 9.5),
        '[5.0-9.0] Bend your knee.',
    )
    self.assertEqual(
        stage0_discover.transcript_excerpt(self._segments(), 3.0, 10.5),
        '[0.0-4.0] Doctor: Hello.\n[5.0-9.0] Bend your knee.\n'
        '[10.0-14.0] Doctor: Thank you.',
    )
    self.assertEqual(
        stage0_discover.transcript_excerpt(self._segments(), 20.0, 30.0), ''
    )

  def test_load_transcript_skips_malformed_entries(self):
    path = self.create_tempfile(
        'case.json',
        content=json.dumps([
            {'start_sec': 10, 'end_sec': 12, 'speaker': 'Patient', 'text': 'B'},
            {'start_sec': 1, 'end_sec': 2, 'text': 'A'},
            {'start_sec': 3, 'text': 'missing end'},
            {'start_sec': 5, 'end_sec': 6, 'text': '   '},
            {'start_sec': 8, 'end_sec': 7, 'text': 'inverted'},
        ]),
    ).full_path
    segments = stage0_discover.load_transcript(path)
    self.assertEqual(
        stage0_discover.format_transcript(segments),
        '[1.0-2.0] A\n[10.0-12.0] Patient: B',
    )

  def test_load_transcript_accepts_segments_object(self):
    path = self.create_tempfile(
        'case.json',
        content=json.dumps(
            {'segments': [{'start_sec': 1, 'end_sec': 2, 'text': 'A'}]}
        ),
    ).full_path
    self.assertLen(stage0_discover.load_transcript(path), 1)


class MakeConfigTest(absltest.TestCase):

  def test_per_mode_defaults(self):
    coarse = stage0_discover.make_config(stage0_discover.COARSE_TO_FINE)
    self.assertEqual(coarse.padding_sec, 5.0)
    self.assertEqual(coarse.max_clip_duration_sec, 60.0)
    single = stage0_discover.make_config(stage0_discover.SINGLE_PASS)
    self.assertEqual(single.padding_sec, 2.0)
    self.assertEqual(single.max_clip_duration_sec, 0.0)

  def test_explicit_values_override_mode_defaults(self):
    single = stage0_discover.make_config(
        stage0_discover.SINGLE_PASS, padding_sec=5.0, max_clip_duration_sec=60.0
    )
    self.assertEqual(single.padding_sec, 5.0)
    self.assertEqual(single.max_clip_duration_sec, 60.0)
    coarse = stage0_discover.make_config(
        stage0_discover.COARSE_TO_FINE, padding_sec=2.0, max_clip_duration_sec=0
    )
    self.assertEqual(coarse.padding_sec, 2.0)
    self.assertEqual(coarse.max_clip_duration_sec, 0.0)

  def test_invalid_settings_raise(self):
    with self.assertRaisesRegex(ValueError, 'fine_fps'):
      stage0_discover.make_config(fine_fps=30.0)
    with self.assertRaisesRegex(ValueError, 'coarse_fps'):
      stage0_discover.make_config(coarse_fps=0.0)
    with self.assertRaisesRegex(ValueError, 'merge_gap_sec'):
      stage0_discover.make_config(merge_gap_sec=-1.0)
    with self.assertRaisesRegex(ValueError, 'Unknown discovery mode'):
      stage0_discover.make_config('two_pass')


class ListVideosTest(absltest.TestCase):

  def test_lists_mp4_files_in_directory(self):
    videos_dir = self.create_tempdir()
    for name in ('b.mp4', 'a.MP4', 'notes.txt'):
      videos_dir.create_file(name)
    videos = stage0_discover.list_videos(videos_dir.full_path, '', 0)
    self.assertEqual([v.video_id for v in videos], ['a', 'b'])
    self.assertLen(stage0_discover.list_videos(videos_dir.full_path, '', 1), 1)

  def test_list_file_takes_precedence(self):
    list_file = self.create_tempfile(
        'videos.txt', content='/data/x/case_01.mp4\n\n/data/y/case_02.mp4\n'
    ).full_path
    videos = stage0_discover.list_videos('/unused', list_file, 0)
    self.assertEqual(
        videos,
        [
            stage0_discover.VideoSource('case_01', '/data/x/case_01.mp4'),
            stage0_discover.VideoSource('case_02', '/data/y/case_02.mp4'),
        ],
    )

  def test_duplicate_video_ids_raise(self):
    list_file = self.create_tempfile(
        'videos.txt', content='/data/x/case_01.mp4\n/data/y/case_01.mp4\n'
    ).full_path
    with self.assertRaisesRegex(ValueError, 'case_01'):
      stage0_discover.list_videos('', list_file, 0)

  def test_missing_input_raises(self):
    with self.assertRaisesRegex(ValueError, 'videos_dir'):
      stage0_discover.list_videos('', '', 0)


class MediaUtilsTest(absltest.TestCase):

  def test_get_media_duration_parses_ffprobe_output(self):
    with mock.patch.object(
        stage0_discover.subprocess,
        'run',
        return_value=subprocess.CompletedProcess([], 0, stdout='123.45\n'),
    ) as mock_run:
      self.assertEqual(
          stage0_discover.get_media_duration_sec('/v/case.mp4'), 123.45
      )
    cmd = mock_run.call_args[0][0]
    self.assertEqual(cmd[0], 'ffprobe')
    self.assertEqual(cmd[-1], '/v/case.mp4')

  def test_get_media_duration_raises_on_bad_output(self):
    with mock.patch.object(
        stage0_discover.subprocess,
        'run',
        return_value=subprocess.CompletedProcess([], 0, stdout='N/A\n'),
    ):
      with self.assertRaises(stage0_discover.MediaError):
        stage0_discover.get_media_duration_sec('/v/case.mp4')

  def test_has_audio_stream(self):
    for stdout, expected in [('1\n', True), ('', False)]:
      with mock.patch.object(
          stage0_discover.subprocess,
          'run',
          return_value=subprocess.CompletedProcess([], 0, stdout=stdout),
      ) as mock_run:
        self.assertEqual(
            stage0_discover.has_audio_stream('/v/case.mp4'), expected
        )
      cmd = mock_run.call_args[0][0]
      self.assertEqual(cmd[0], 'ffprobe')
      self.assertEqual(cmd[cmd.index('-select_streams') + 1], 'a')
      self.assertEqual(cmd[-1], '/v/case.mp4')

  def test_ffmpeg_failure_raises_media_error(self):
    error = subprocess.CalledProcessError(1, ['ffmpeg'], stderr=b'bad input')
    with mock.patch.object(
        stage0_discover.subprocess, 'run', side_effect=error
    ):
      with self.assertRaisesRegex(stage0_discover.MediaError, 'bad input'):
        stage0_discover.cut_video('/v.mp4', 1.0, 2.0, '/out.mp4', False)

  def test_cut_video_arguments_and_output_check(self):
    out_path = os.path.join(self.create_tempdir().full_path, 'clip.mp4')

    def _write_tiny_file(args):
      with open(args[-1], 'wb') as f:
        f.write(b'tiny')

    with mock.patch.object(
        stage0_discover, '_run_ffmpeg', side_effect=_write_tiny_file
    ) as mock_ffmpeg:
      with self.assertRaisesRegex(stage0_discover.MediaError, 'too small'):
        stage0_discover.cut_video('/v.mp4', 12.5, 20.0, out_path, False)
    args = mock_ffmpeg.call_args[0][0]
    self.assertEqual(args[:6], ['-ss', '12.500', '-i', '/v.mp4', '-t', '7.500'])
    self.assertIn('-an', args)
    self.assertIn('libx264', args)


class DiscoveryPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.videos_dir = self.create_tempdir('videos')
    self.clips_dir = self.create_tempdir('clips').full_path
    self.transcripts_dir = self.create_tempdir('transcripts')
    self.video = self._make_video('case_01')
    self.ffmpeg_calls = []
    self.enter_context(
        mock.patch.object(
            stage0_discover, '_run_ffmpeg', side_effect=self._fake_ffmpeg
        )
    )
    self.enter_context(
        mock.patch.object(
            stage0_discover, 'get_media_duration_sec', return_value=120.0
        )
    )
    self.mock_has_audio = self.enter_context(
        mock.patch.object(
            stage0_discover, 'has_audio_stream', return_value=True
        )
    )

  def _make_video(self, video_id):
    path = self.videos_dir.create_file(
        f'{video_id}.mp4', content='source-video-' * 400
    ).full_path
    return stage0_discover.VideoSource(video_id, path)

  def _fake_ffmpeg(self, args):
    """Records the call and writes a non-trivial output file."""
    self.ffmpeg_calls.append(list(args))
    with open(args[-1], 'wb') as f:
      f.write(b'\0' * 2048)

  def _write_transcript(self, video_id, segments):
    self.transcripts_dir.create_file(
        f'{video_id}.json', content=json.dumps(segments)
    )

  def test_coarse_to_fine_with_provided_transcript(self):
    self._write_transcript(
        'case_01',
        [
            {
                'start_sec': 8,
                'end_sec': 12,
                'speaker': 'Doctor',
                'text': 'Bend your left knee.',
            },
            {
                'start_sec': 50,
                'end_sec': 55,
                'speaker': 'Patient',
                'text': 'It hurts here.',
            },
            {'start_sec': 100, 'end_sec': 104, 'text': 'Thank you.'},
        ],
    )
    lm = FakeLanguageModel({
        stage0_discover.LanguageScanResult: [
            stage0_discover.LanguageScanResult([
                _LanguageWindow(10, 15, 'flexion', ['action'], 'Bend.'),
                _LanguageWindow(50, 55, 'pointing', ['SPATIAL'], 'Here.'),
            ])
        ],
        stage0_discover.VisualScanResult: [
            stage0_discover.VisualScanResult([
                _VisualWindow(12, 20, 'flexing', ['ACTION', 'TEMPORAL']),
                _VisualWindow(80, 85, 'grimace', ['APPEARANCE']),
            ])
        ],
        stage0_discover.FineScanResult: [
            stage0_discover.FineScanResult([_event(6.0, 9.0)]),
            stage0_discover.FineScanResult(
                [_event(5.0, 8.0, 'SPATIAL', action='pointing')]
            ),
            stage0_discover.FineScanResult([]),
        ],
    })
    config = stage0_discover.make_config(stage0_discover.COARSE_TO_FINE)

    record, clips = stage0_discover.process_video(
        self.video, lm, config, self.transcripts_dir.full_path, self.clips_dir
    )

    self.assertEqual(record['status'], 'success', record)
    self.assertEqual(record['transcript_source'], 'provided')
    self.assertEqual(
        record['counts'],
        {
            'transcript_segments': 3,
            'pass1a_windows': 2,
            'pass1b_windows': 2,
            'merged_windows': 3,
            'window_errors': 0,
            'pre_dedup_events': 2,
            'post_dedup_events': 2,
            'skipped_short_events': 0,
            'truncated_events': 0,
            'clip_errors': 0,
            'clips': 2,
        },
    )
    self.assertEqual(
        [w['sources'] for w in record['windows']],
        [['TRANSCRIPT', 'VISUAL'], ['TRANSCRIPT'], ['VISUAL']],
    )
    self.assertEmpty(lm.contents_for(stage0_discover.TranscriptionResult))

    (language_contents,) = lm.contents_for(stage0_discover.LanguageScanResult)
    self.assertLen(language_contents, 1)
    self.assertIn(
        '[8.0-12.0] Doctor: Bend your left knee.', language_contents[0]
    )

    (visual_contents,) = lm.contents_for(stage0_discover.VisualScanResult)
    self.assertEqual(visual_contents[0]['video_metadata'], {'fps': 5.0})
    self.assertEqual(
        visual_contents[0]['inline_data']['mime_type'], 'video/mp4'
    )
    self.assertIn(['-c:v', 'copy', '-an'], [c[4:7] for c in self.ffmpeg_calls])

    fine_contents = lm.contents_for(stage0_discover.FineScanResult)
    self.assertLen(fine_contents, 3)
    for contents in fine_contents:
      self.assertEqual(contents[0]['video_metadata'], {'fps': 10.0})
    self.assertIn('from 5.0s to 25.0s', fine_contents[0][1])
    self.assertIn('Bend your left knee.', fine_contents[0][1])
    self.assertNotIn('Thank you.', fine_contents[0][1])
    self.assertIn('No transcript available.', fine_contents[2][1])
    window_cuts = [c for c in self.ffmpeg_calls if 'window_000' in c[-1]]
    self.assertEqual(
        window_cuts[0][:6],
        ['-ss', '5.000', '-i', self.video.path, '-t', '20.000'],
    )
    self.assertIn('-an', window_cuts[0])

    self.assertLen(clips, 2)
    first = clips[0]
    for field in _STAGE1_CLIP_FIELDS:
      self.assertIn(field, first)
    self.assertEqual(first['clip_file'], 'case_01_stage0_clip_01.mp4')
    self.assertEqual(first['source_video_id'], 'case_01')
    self.assertEqual(
        (first['event_start_sec'], first['event_end_sec']), (11.0, 14.0)
    )
    self.assertEqual((first['start_sec'], first['end_sec']), (6.0, 19.0))
    self.assertEqual(first['clip_duration_sec'], 13.0)
    self.assertEqual(first['primary_eim_facet'], 'ACTION')
    self.assertEqual(first['window_sources'], ['TRANSCRIPT', 'VISUAL'])
    self.assertEqual(
        first['transcript'], '[8.0-12.0] Doctor: Bend your left knee.'
    )
    for key in ('clip_path', 'clip_path_noaudio', 'clip_path_audio'):
      self.assertTrue(os.path.exists(first[key]), first[key])
    self.assertEqual(clips[1]['primary_eim_facet'], 'SPATIAL')
    self.assertEqual(clips[1]['event_start_sec'], 50.0)
    self.assertIn('It hurts here.', clips[1]['transcript'])

  def test_coarse_to_fine_transcribes_audio_without_transcript(self):
    lm = FakeLanguageModel({
        stage0_discover.TranscriptionResult: [
            stage0_discover.TranscriptionResult([
                stage0_discover.TranscriptSegment(
                    3.0, 6.0, 'Now press on the right shin.', 'Narrator'
                )
            ])
        ],
        stage0_discover.LanguageScanResult: [
            stage0_discover.LanguageScanResult(
                [_LanguageWindow(3, 8, 'palpation', ['SPATIAL'], 'Press.')]
            )
        ],
        stage0_discover.VisualScanResult: [
            stage0_discover.VisualScanResult([])
        ],
        stage0_discover.FineScanResult: [
            stage0_discover.FineScanResult(
                [_event(4.0, 9.0, 'SPATIAL', 'right_shin', 'palpating')]
            )
        ],
    })
    config = stage0_discover.make_config(stage0_discover.COARSE_TO_FINE)

    record, clips = stage0_discover.process_video(
        self.video, lm, config, '', self.clips_dir
    )

    self.assertEqual(record['status'], 'success', record)
    self.assertEqual(record['transcript_source'], 'gemini_transcription')
    (transcription_contents,) = lm.contents_for(
        stage0_discover.TranscriptionResult
    )
    audio_part = transcription_contents[0]
    self.assertEqual(audio_part['inline_data']['mime_type'], 'audio/wav')
    self.assertNotIn('video_metadata', audio_part)
    self.assertTrue(any('16000' in call for call in self.ffmpeg_calls))
    (language_contents,) = lm.contents_for(stage0_discover.LanguageScanResult)
    self.assertIn(
        'Narrator: Now press on the right shin.', language_contents[0]
    )
    self.assertLen(clips, 1)
    self.assertIn('right shin', clips[0]['transcript'])

  def test_visual_scan_only_without_transcript_or_transcription(self):
    lm = FakeLanguageModel({
        stage0_discover.VisualScanResult: [
            stage0_discover.VisualScanResult(
                [_VisualWindow(30, 35, 'flexing', ['ACTION'])]
            )
        ],
        stage0_discover.FineScanResult: [
            stage0_discover.FineScanResult([_event(5.0, 10.0)])
        ],
    })
    config = stage0_discover.make_config(
        stage0_discover.COARSE_TO_FINE, transcribe_audio=False
    )

    record, clips = stage0_discover.process_video(
        self.video, lm, config, '', self.clips_dir
    )

    self.assertEqual(record['status'], 'success', record)
    self.assertEqual(record['transcript_source'], 'none')
    self.assertEqual(record['counts']['pass1a_windows'], 0)
    self.assertEqual(record['counts']['pass1b_windows'], 1)
    self.assertEqual(
        [cls for cls, _ in lm.calls],
        [stage0_discover.VisualScanResult, stage0_discover.FineScanResult],
    )
    (fine_contents,) = lm.contents_for(stage0_discover.FineScanResult)
    self.assertIn('No transcript available.', fine_contents[1])
    self.assertEqual(clips[0]['transcript'], '')

  def test_failed_window_does_not_fail_the_video(self):
    lm = FakeLanguageModel({
        stage0_discover.VisualScanResult: [
            stage0_discover.VisualScanResult([
                _VisualWindow(10, 15, 'flexing', ['ACTION']),
                _VisualWindow(60, 65, 'pointing', ['SPATIAL']),
            ])
        ],
        stage0_discover.FineScanResult: [
            ValueError('Failed to parse response'),
            stage0_discover.FineScanResult([_event(5.0, 10.0)]),
        ],
    })
    config = stage0_discover.make_config(
        stage0_discover.COARSE_TO_FINE, transcribe_audio=False
    )

    record, clips = stage0_discover.process_video(
        self.video, lm, config, '', self.clips_dir
    )

    self.assertEqual(record['status'], 'success', record)
    self.assertEqual(record['counts']['window_errors'], 1)
    self.assertStartsWith(record['errors'][0], 'pass2_window_0: ValueError')
    self.assertLen(clips, 1)
    self.assertEqual(clips[0]['event_start_sec'], 60.0)

  def test_all_windows_failing_marks_video_as_error(self):
    lm = FakeLanguageModel({
        stage0_discover.VisualScanResult: [
            stage0_discover.VisualScanResult(
                [_VisualWindow(10, 15, 'flexing', ['ACTION'])]
            )
        ],
        stage0_discover.FineScanResult: [RuntimeError('quota exceeded')],
    })
    config = stage0_discover.make_config(
        stage0_discover.COARSE_TO_FINE, transcribe_audio=False
    )

    record, clips = stage0_discover.process_video(
        self.video, lm, config, '', self.clips_dir
    )

    self.assertEqual(record['status'], 'error')
    self.assertIn('All 1 fine-scan windows failed', record['error'])
    self.assertEmpty(clips)

  def test_no_candidate_windows_gives_no_events(self):
    lm = FakeLanguageModel({
        stage0_discover.VisualScanResult: [
            stage0_discover.VisualScanResult([])
        ],
    })
    config = stage0_discover.make_config(
        stage0_discover.COARSE_TO_FINE, transcribe_audio=False
    )

    record, clips = stage0_discover.process_video(
        self.video, lm, config, '', self.clips_dir
    )

    self.assertEqual(record['status'], 'no_events')
    self.assertEqual(record['counts']['merged_windows'], 0)
    self.assertEmpty(clips)

  def test_single_pass_uses_one_call_on_full_video_with_audio(self):
    discovered = stage0_discover.DiscoveredEIMEvent
    lm = FakeLanguageModel({
        stage0_discover.VideoDiscoveryResult: [
            stage0_discover.VideoDiscoveryResult([
                discovered(
                    10,
                    20,
                    'action',
                    'Knee extension test.',
                    'Knee extension',
                    'left knee',
                    'Extend your knee fully.',
                ),
                # Too short after padding (28.0-32.5 s).
                discovered(30, 30.5, 'APPEARANCE', 'Grimace.', 'G', 'face', ''),
                # Long, but single_pass has no maximum clip duration.
                discovered(40, 110, 'TEMPORAL', 'Gait.', 'Gait', 'legs', ''),
            ])
        ],
    })
    config = stage0_discover.make_config(stage0_discover.SINGLE_PASS)

    record, clips = stage0_discover.process_video(
        self.video, lm, config, '', self.clips_dir
    )

    self.assertEqual(record['status'], 'success', record)
    self.assertEqual(record['transcript_source'], 'speech_summary')
    self.assertLen(lm.calls, 1)
    (contents,) = lm.contents_for(stage0_discover.VideoDiscoveryResult)
    self.assertEqual(contents[0]['video_metadata'], {'fps': 5.0})
    with open(self.video.path, 'rb') as f:
      self.assertEqual(contents[0]['inline_data']['data'], f.read())
    self.assertIn('Exam Information Moments', contents[1])

    self.assertEqual(record['counts']['skipped_short_events'], 1)
    self.assertEqual(record['counts']['truncated_events'], 0)
    self.assertLen(clips, 2)
    first, second = clips
    self.assertEqual((first['start_sec'], first['end_sec']), (8.0, 22.0))
    self.assertEqual(first['padding_sec'], 2.0)
    self.assertEqual(first['primary_eim_facet'], 'ACTION')
    self.assertEqual(first['transcript'], 'Extend your knee fully.')
    self.assertEqual(first['maneuver_name'], 'Knee extension')
    self.assertEqual(first['anatomical_target'], 'left knee')
    self.assertEqual(second['clip_file'], 'case_01_stage0_clip_02.mp4')
    self.assertEqual(second['clip_duration_sec'], 74.0)
    self.assertEqual(second['transcript'], 'No speech detected.')
    first_cut = self.ffmpeg_calls[0]
    self.assertEqual(
        first_cut[:6], ['-ss', '8.000', '-i', self.video.path, '-t', '14.000']
    )
    self.assertIn('0:a:0?', first_cut)

  def test_clip_with_failed_audio_output_is_skipped(self):

    def _ffmpeg_without_audio(args):
      self.ffmpeg_calls.append(list(args))
      with open(args[-1], 'wb') as f:
        f.write(b'\0' * (10 if args[-1].endswith('_audio.wav') else 2048))

    lm = FakeLanguageModel({
        stage0_discover.VideoDiscoveryResult: [
            stage0_discover.VideoDiscoveryResult([
                stage0_discover.DiscoveredEIMEvent(
                    10, 20, 'ACTION', 'Test.', 'Test', 'knee', 'Speech.'
                )
            ])
        ],
    })
    config = stage0_discover.make_config(stage0_discover.SINGLE_PASS)

    with mock.patch.object(
        stage0_discover, '_run_ffmpeg', side_effect=_ffmpeg_without_audio
    ):
      record, clips = stage0_discover.process_video(
          self.video, lm, config, '', self.clips_dir
      )

    self.assertEqual(record['status'], 'no_events')
    self.assertEqual(record['counts']['clip_errors'], 1)
    self.assertEmpty(clips)
    self.assertEmpty(os.listdir(self.clips_dir))

  def test_clip_of_silent_video_is_kept_without_audio_file(self):
    self.mock_has_audio.return_value = False
    lm = FakeLanguageModel({
        stage0_discover.VideoDiscoveryResult: [
            stage0_discover.VideoDiscoveryResult([
                stage0_discover.DiscoveredEIMEvent(
                    10, 20, 'ACTION', 'Test.', 'Test', 'knee', 'Speech.'
                )
            ])
        ],
    })
    config = stage0_discover.make_config(stage0_discover.SINGLE_PASS)

    record, clips = stage0_discover.process_video(
        self.video, lm, config, '', self.clips_dir
    )

    self.assertEqual(record['status'], 'success')
    self.assertEqual(record['counts']['clip_errors'], 0)
    (clip,) = clips
    self.assertEqual(clip['clip_path_audio'], '')
    self.assertTrue(os.path.exists(clip['clip_path']))
    self.assertTrue(os.path.exists(clip['clip_path_noaudio']))
    self.assertCountEqual(
        os.listdir(self.clips_dir),
        ['case_01_stage0_clip_01.mp4', 'case_01_stage0_clip_01_noaudio.mp4'],
    )

  def test_unreadable_video_is_recorded_as_error(self):
    lm = FakeLanguageModel({})
    config = stage0_discover.make_config()
    with mock.patch.object(
        stage0_discover,
        'get_media_duration_sec',
        side_effect=stage0_discover.MediaError('cannot read'),
    ):
      record, clips = stage0_discover.process_video(
          self.video, lm, config, '', self.clips_dir
      )
    self.assertEqual(record['status'], 'error')
    self.assertIn('cannot read', record['error'])
    self.assertEmpty(clips)

  def test_run_discovery_writes_manifest_and_resumes(self):
    videos = [self.video, self._make_video('case_02')]
    manifest_path = os.path.join(self.create_tempdir().full_path, 'm.json')
    config = stage0_discover.make_config(
        stage0_discover.COARSE_TO_FINE, transcribe_audio=False
    )

    def _responses(visual_results):
      return {
          stage0_discover.VisualScanResult: visual_results,
          stage0_discover.FineScanResult: [
              stage0_discover.FineScanResult([_event(5.0, 10.0)])
          ],
      }

    window = [_VisualWindow(30, 35, 'flexing', ['ACTION'])]
    first_lm = FakeLanguageModel(
        _responses([
            stage0_discover.VisualScanResult(window),
            RuntimeError('service unavailable'),
        ])
    )
    manifest = stage0_discover.run_discovery(
        videos,
        first_lm,
        config,
        '',
        self.clips_dir,
        manifest_path,
        model_name='test-model',
        num_workers=1,
    )

    with open(manifest_path, 'r') as f:
      self.assertEqual(json.load(f), manifest)
    self.assertEqual(manifest['stage'], 'stage0_discovery')
    self.assertEqual(manifest['discovery_mode'], 'coarse_to_fine')
    self.assertEqual(manifest['model'], 'test-model')
    self.assertEqual(manifest['coarse_fps'], 5.0)
    self.assertEqual(manifest['fine_fps'], 10.0)
    self.assertNotIn('single_pass_fps', manifest)
    self.assertEqual(manifest['settings']['padding_sec'], 5.0)
    self.assertEqual(manifest['settings']['max_clip_duration_sec'], 60.0)
    self.assertEqual(manifest['num_videos'], 2)
    self.assertEqual(manifest['num_clips'], 1)
    self.assertEqual(
        manifest['video_status_counts'], {'success': 1, 'error': 1}
    )
    self.assertEqual(
        [v['video_id'] for v in manifest['videos']], ['case_01', 'case_02']
    )
    self.assertIn('No coarse scan succeeded', manifest['videos'][1]['error'])
    for field in _STAGE1_CLIP_FIELDS:
      self.assertIn(field, manifest['clips'][0])

    # A second run only retries the video that failed.
    second_lm = FakeLanguageModel(
        _responses([stage0_discover.VisualScanResult(window)])
    )
    manifest = stage0_discover.run_discovery(
        videos,
        second_lm,
        config,
        '',
        self.clips_dir,
        manifest_path,
        model_name='test-model',
        num_workers=1,
    )
    self.assertLen(second_lm.calls, 2)
    self.assertEqual(manifest['video_status_counts'], {'success': 2})
    self.assertEqual(
        [c['source_video_id'] for c in manifest['clips']],
        ['case_01', 'case_02'],
    )

    with self.assertRaisesRegex(ValueError, 'discovery_mode'):
      stage0_discover.run_discovery(
          videos,
          second_lm,
          stage0_discover.make_config(stage0_discover.SINGLE_PASS),
          '',
          self.clips_dir,
          manifest_path,
          model_name='test-model',
      )


if __name__ == '__main__':
  absltest.main()
