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
"""Unit tests for Stage 1 evidence extraction utilities."""

import json
import os
import subprocess
from unittest import mock

from absl.testing import absltest
import data_models
import stage1_prototype
import template_utils


class Stage1PrototypeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.eim_facets = stage1_prototype._load_eim_facets()

  def test_load_eim_facets_structure(self):
    required_facets = {'action', 'spatial', 'temporal', 'appearance', 'absence'}
    self.assertTrue(required_facets.issubset(self.eim_facets.keys()))
    for key in required_facets:
      self.assertIn('label', self.eim_facets[key])
      self.assertIn('description', self.eim_facets[key])
      self.assertNotEmpty(self.eim_facets[key]['label'])
      self.assertNotEmpty(self.eim_facets[key]['description'])

  def test_parse_clip_name_valid(self):
    self.assertEqual(
        stage1_prototype._parse_clip_name('10.0_indicator_0_window_0.mp4'),
        ('10.0', 0),
    )
    self.assertEqual(
        stage1_prototype._parse_clip_name('123.0_indicator_2_window_1.mp4'),
        ('123.0', 2),
    )

  def test_parse_clip_name_invalid(self):
    self.assertIsNone(
        stage1_prototype._parse_clip_name('invalid_clip_format.mp4')
    )
    self.assertIsNone(
        stage1_prototype._parse_clip_name('10.0_no_indicator.mp4')
    )
    self.assertIsNone(stage1_prototype._parse_clip_name(''))

  def test_render_pass1_template_valid(self):
    prompt = template_utils.render_template(
        'stage1_pass1_blind',
        facets=self.eim_facets,
    )
    self.assertIn('You are a senior attending physician', prompt)
    self.assertIn('ACTION', prompt)
    self.assertIn('SPATIAL', prompt)
    self.assertIn('TEMPORAL', prompt)
    self.assertIn('APPEARANCE', prompt)
    self.assertIn('ABSENCE', prompt)

  def test_render_pass2_template_valid(self):
    observations = [
        {
            'label': 'ACTION [EIM: Intentionality]',
            'value': 'Patient rubs right shoulder',
        },
        {'label': 'SPATIAL [EIM: Space]', 'value': 'Right anterior shoulder'},
        {'label': 'TEMPORAL [EIM: Time]', 'value': '00:05-00:10'},
        {
            'label': 'APPEARANCE [EIM: Entity]',
            'value': 'Grimacing with eyes squinted',
        },
        {
            'label': 'ABSENCE [EIM: Causation]',
            'value': 'No accessory muscle use',
        },
    ]
    prompt = template_utils.render_template(
        'stage1_pass2_grounded',
        primary_sign='Right shoulder rubbing',
        observations=observations,
        condition='Rotator Cuff Tear',
        symptom='Shoulder pain',
        transcript='[0.0-10.0] Doctor: Where is the pain?',
    )
    self.assertIn('Rotator Cuff Tear', prompt)
    self.assertIn('Right shoulder rubbing', prompt)
    self.assertIn('SUPPORTED', prompt)
    self.assertIn('NOT_MENTIONED', prompt)

  def test_get_transcript_segment_overlapping_and_padded(self):
    turns = [
        data_models.TranscriptTurn(
            speaker='Doctor',
            text='Hello, how are you feeling today?',
            start_time=0.0,
            end_time=5.0,
            medically_significant=False,
        ),
        data_models.TranscriptTurn(
            speaker='Patient',
            text='I have sharp pain in my right shoulder.',
            start_time=10.0,
            end_time=15.0,
            medically_significant=True,
        ),
        data_models.TranscriptTurn(
            speaker='Doctor',
            text='Let me check your range of motion.',
            start_time=40.0,
            end_time=45.0,
            medically_significant=False,
        ),
    ]
    case = data_models.AvEvaluationCase(
        case_id='101',
        case_condition='Rotator Cuff Tear',
        file_name='case_0101.json',
        summary_keywords=['shoulder', 'pain'],
        key_medical_information='Shoulder pain on abduction',
        audio_video_indicators=data_models.AudioVideoIndicatorList(
            av_indicators=[
                data_models.AudioVideoIndicator(
                    av_indicator='Rubbing shoulder',
                    is_continuous=False,
                    time_windows=[(10.0, 15.0)],
                )
            ]
        ),
        raw_video_path='',
        raw_transcript_path='',
        transcript=data_models.Transcript(turns=turns),
    )

    # Indicator window is 10.0-15.0 with padding 15s -> window 0.0-30.0
    segment = stage1_prototype._get_transcript_segment(case, 0, padding_s=15.0)
    self.assertIn('Hello, how are you feeling today?', segment)
    self.assertIn('sharp pain in my right shoulder', segment)
    self.assertNotIn('range of motion', segment)

  def test_get_transcript_segment_out_of_bounds(self):
    case = data_models.AvEvaluationCase(
        case_id='101',
        case_condition='None',
        file_name='case_0101.json',
        summary_keywords=[],
        key_medical_information='',
        audio_video_indicators=data_models.AudioVideoIndicatorList(
            av_indicators=[]
        ),
        raw_video_path='',
        raw_transcript_path='',
        transcript=data_models.Transcript(turns=[]),
    )
    segment = stage1_prototype._get_transcript_segment(case, 5)
    self.assertEqual(segment, '')


class GetClipDurationTest(absltest.TestCase):

  def test_reads_duration_with_ffprobe(self):
    with mock.patch.object(
        stage1_prototype.subprocess,
        'run',
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout='12.5\n'
        ),
    ) as mock_run:
      duration = stage1_prototype._get_clip_duration_sec('/clips/clip.mp4')

    self.assertEqual(duration, 12.5)
    cmd = mock_run.call_args.args[0]
    self.assertEqual(cmd[0], 'ffprobe')
    self.assertEqual(cmd[-1], '/clips/clip.mp4')

  def test_falls_back_to_default_if_ffprobe_fails(self):
    with mock.patch.object(
        stage1_prototype.subprocess,
        'run',
        side_effect=FileNotFoundError('ffprobe'),
    ):
      duration = stage1_prototype._get_clip_duration_sec('/clips/clip.mp4')

    self.assertEqual(duration, stage1_prototype._DEFAULT_CLIP_DURATION_SEC)


class Stage0ManifestTest(absltest.TestCase):
  """Tests reading clips from a Stage 0 manifest (stage0_discover.py)."""

  def setUp(self):
    super().setUp()
    self.eim_facets = stage1_prototype._load_eim_facets()
    clips_dir = self.create_tempdir()
    self.clip_path = clips_dir.create_file(
        'vid_stage0_clip_01.mp4', content='av-clip'
    ).full_path
    self.noaudio_path = clips_dir.create_file(
        'vid_stage0_clip_01_noaudio.mp4', content='muted-clip'
    ).full_path
    # A clip entry as written by stage0_discover.py.
    self.entry = {
        'clip_id': 'vid_stage0_clip_01',
        'clip_file': 'vid_stage0_clip_01.mp4',
        'clip_path': self.clip_path,
        'clip_path_noaudio': self.noaudio_path,
        'clip_path_audio': self.clip_path.replace('.mp4', '_audio.wav'),
        'source_video_id': 'vid',
        'discovery_mode': 'coarse_to_fine',
        'start_sec': 6.0,
        'end_sec': 19.0,
        'clip_duration_sec': 13.0,
        'event_start_sec': 11.0,
        'event_end_sec': 14.0,
        'primary_eim_facet': 'ACTION',
        'clinical_event_description': 'Patient flexes the left knee.',
        'transcript': '[8.0-12.0] Doctor: Bend your left knee.',
    }

  def _fake_lm(self):
    lm = mock.MagicMock()
    blind = stage1_prototype.BlindObservation(
        action_observations='Flexes the knee.',
        spatial_observations='Left knee.',
        temporal_observations='00:05-00:08 knee flexion.',
        appearance_observations='Neutral expression.',
        absence_observations='No swelling.',
        primary_sign='Left knee flexion.',
    )
    batch = stage1_prototype.FacetVerificationBatch(
        facet_verdicts=[
            stage1_prototype.FacetVerification(
                eim_facet='action',
                transcript_verdict='SUPPORTED',
                transcript_evidence='Bend your left knee.',
            )
        ]
    )
    lm.generate_thought_and_object.side_effect = [(blind, ''), (batch, '')]
    return lm

  def test_load_stage0_clips(self):
    manifest = self.create_tempfile(
        'manifest.json',
        content=json.dumps(
            {'stage': 'stage0_discovery', 'clips': [self.entry]}
        ),
    ).full_path
    self.assertEqual(
        stage1_prototype._load_stage0_clips(manifest), [self.entry]
    )
    clip_list = self.create_tempfile(
        'clips.json', content=json.dumps([self.entry])
    ).full_path
    self.assertEqual(
        stage1_prototype._load_stage0_clips(clip_list), [self.entry]
    )

  def test_process_stage0_clip_uses_manifest_fields(self):
    lm = self._fake_lm()
    with mock.patch.object(
        stage1_prototype,
        '_get_clip_duration_sec',
        side_effect=AssertionError('The manifest duration should be used.'),
    ):
      result = stage1_prototype._process_stage0_clip(
          self.entry, lm, self.eim_facets
      )

    self.assertEqual(result['status'], 'success')
    self.assertEqual(result['clip'], 'vid_stage0_clip_01.mp4')
    self.assertEqual(result['case_id'], 'vid')
    self.assertIsNone(result['indicator_idx'])
    self.assertEqual(result['condition'], 'General Clinical')
    self.assertEqual(result['symptom'], 'Patient flexes the left knee.')
    self.assertEqual(result['clip_duration_sec'], 13.0)
    self.assertEqual(result['clip_duration_str'], '00:13')
    self.assertEqual(result['evidence']['primary_sign'], 'Left knee flexion.')
    self.assertEqual(result['facet_verdicts']['ACTION']['verdict'], 'SUPPORTED')
    self.assertEqual(result['stage0']['source_video_id'], 'vid')
    self.assertEqual(result['stage0']['primary_eim_facet'], 'ACTION')
    pass1_contents = lm.generate_thought_and_object.call_args_list[0][0][0]
    self.assertEqual(pass1_contents[0]['inline_data']['data'], b'muted-clip')
    pass2_prompt = lm.generate_thought_and_object.call_args_list[1][0][0]
    self.assertIn('[8.0-12.0] Doctor: Bend your left knee.', pass2_prompt)
    self.assertIn('Annotated indicator: "Patient flexes the left knee."',
                  pass2_prompt)

  def test_process_stage0_clip_passes_fps_to_video_part(self):
    lm = self._fake_lm()
    result = stage1_prototype._process_stage0_clip(
        self.entry, lm, self.eim_facets, fps=5.0
    )
    self.assertEqual(result['status'], 'success')
    pass1_contents = lm.generate_thought_and_object.call_args_list[0][0][0]
    self.assertEqual(pass1_contents[0]['video_metadata'], {'fps': 5.0})

  def test_process_stage0_clip_falls_back_to_clip_with_audio(self):
    os.remove(self.noaudio_path)
    lm = self._fake_lm()
    result = stage1_prototype._process_stage0_clip(
        self.entry, lm, self.eim_facets
    )
    self.assertEqual(result['status'], 'success')
    pass1_contents = lm.generate_thought_and_object.call_args_list[0][0][0]
    self.assertEqual(pass1_contents[0]['inline_data']['data'], b'av-clip')

  def test_process_stage0_clip_with_missing_files_returns_none(self):
    entry = dict(
        self.entry,
        clip_path='/missing/clip.mp4',
        clip_path_noaudio='/missing/clip_noaudio.mp4',
    )
    self.assertIsNone(
        stage1_prototype._process_stage0_clip(
            entry, mock.MagicMock(), self.eim_facets
        )
    )

  def test_extraction_pipeline_reads_stage0_clips_and_resumes(self):
    output_file = os.path.join(self.create_tempdir().full_path, 'stage1.json')
    lm = self._fake_lm()

    def _run():
      return stage1_prototype._run_extraction_pipeline(
          clips=[],
          case_annotations={},
          cases_dir='',
          clips_dir='',
          lm=lm,
          eim_facets=self.eim_facets,
          output_file=output_file,
          model_name='test-model',
          stage0_clips=[self.entry],
      )

    results = _run()
    self.assertLen(results, 1)
    self.assertEqual(results[0]['clip'], 'vid_stage0_clip_01.mp4')
    with open(output_file, 'r') as f:
      self.assertEqual(json.load(f)['source'], 'stage0_manifest')
    self.assertLen(_run(), 1)  # Resumed from the checkpoint.
    self.assertEqual(lm.generate_thought_and_object.call_count, 2)


if __name__ == '__main__':
  absltest.main()
