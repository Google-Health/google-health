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
"""Unit tests for the MCQ evaluator."""

import os
import subprocess
from unittest import mock

from absl.testing import absltest

import eval_mcq


class EvaluateSingleItemTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    clips_dir = self.create_tempdir()
    self.clips_dir = clips_dir.full_path
    self.clip_path = clips_dir.create_file(
        'clip_01.mp4', content='video-bytes'
    ).full_path
    self.item = {
        'clip_file': 'clip_01.mp4',
        'question': 'Which knee does the patient touch?',
        'correct_answer': 'Left knee',
        'distractors': ['Right knee', 'Neither knee'],
        'question_type': 'spatial',
    }
    _, correct_key, _ = eval_mcq._format_question_options(self.item)
    self.lm = mock.MagicMock()
    self.lm.generate_thought_and_object.return_value = (
        eval_mcq.MCQAnswer(
            reasoning='Touches the left knee.',
            selected_option=correct_key,
            confidence='HIGH',
        ),
        '',
    )

  def _evaluate(self, include_audio):
    return eval_mcq._evaluate_single_item(
        item=self.item,
        lm=self.lm,
        clips_dir=self.clips_dir,
        include_audio=include_audio,
        text_only=False,
        audio_only=False,
        fps=5.0,
    )

  def test_video_and_audio_sends_clip_at_requested_fps(self):
    with mock.patch.object(eval_mcq.subprocess, 'run') as mock_run:
      result = self._evaluate(include_audio=True)

    self.assertTrue(result['is_correct'])
    mock_run.assert_not_called()
    contents = self.lm.generate_thought_and_object.call_args[0][0]
    self.assertEqual(contents[0]['video_metadata'], {'fps': 5.0})
    self.assertEqual(contents[0]['inline_data']['data'], b'video-bytes')

  def test_video_only_sends_clip_without_audio(self):
    ffmpeg_cmds = []

    def _fake_ffmpeg(cmd, **unused_kwargs):
      ffmpeg_cmds.append(cmd)
      with open(cmd[-1], 'wb') as f:
        f.write(b'muted-video-bytes')

    with mock.patch.object(
        eval_mcq.subprocess, 'run', side_effect=_fake_ffmpeg
    ):
      result = self._evaluate(include_audio=False)

    self.assertTrue(result['is_correct'])
    (cmd,) = ffmpeg_cmds
    self.assertEqual(cmd[cmd.index('-i') + 1], self.clip_path)
    self.assertIn('-an', cmd)
    self.assertEqual(cmd[cmd.index('-c:v') + 1], 'copy')
    contents = self.lm.generate_thought_and_object.call_args[0][0]
    self.assertEqual(contents[0]['inline_data']['data'], b'muted-video-bytes')
    self.assertEqual(contents[0]['video_metadata'], {'fps': 5.0})
    self.assertFalse(os.path.exists(cmd[-1]))

  def test_video_only_records_error_if_audio_cannot_be_removed(self):
    with mock.patch.object(
        eval_mcq.subprocess,
        'run',
        side_effect=subprocess.CalledProcessError(1, ['ffmpeg']),
    ):
      result = self._evaluate(include_audio=False)

    self.assertEqual(result['status'], 'error')
    self.lm.generate_thought_and_object.assert_not_called()


class AudioOnlyEvaluationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    clips_dir = self.create_tempdir()
    self.clips_dir = clips_dir.full_path
    self.clip_path = clips_dir.create_file(
        'clip_01.mp4', content='video-bytes'
    ).full_path
    self.item = {
        'clip_file': 'clip_01.mp4',
        'question': 'Which knee does the patient say hurts?',
        'correct_answer': 'Left knee',
        'distractors': ['Right knee', 'Neither knee'],
        'question_type': 'spatial',
    }
    self.ffmpeg_cmds = []

  def _fake_ffmpeg(self, cmd, **unused_kwargs):
    self.ffmpeg_cmds.append(cmd)
    with open(cmd[-1], 'wb') as f:
      f.write(b'audio-bytes')

  def _evaluate(self, lm):
    with mock.patch.object(
        eval_mcq.subprocess, 'run', side_effect=self._fake_ffmpeg
    ):
      return eval_mcq._evaluate_single_item(
          item=self.item,
          lm=lm,
          clips_dir=self.clips_dir,
          include_audio=False,
          text_only=False,
          audio_only=True,
      )

  def test_extracts_audio_from_clip_and_removes_temp_file(self):
    _, correct_key, _ = eval_mcq._format_question_options(self.item)
    lm = mock.MagicMock()
    lm.generate_thought_and_object.return_value = (
        eval_mcq.MCQAnswer(
            reasoning='The patient says the left knee hurts.',
            selected_option=correct_key,
            confidence='HIGH',
        ),
        '',
    )

    result = self._evaluate(lm)

    self.assertTrue(result['is_correct'])
    self.assertLen(self.ffmpeg_cmds, 1)
    cmd = self.ffmpeg_cmds[0]
    self.assertEqual(cmd[cmd.index('-i') + 1], self.clip_path)
    contents = lm.generate_thought_and_object.call_args[0][0]
    self.assertEqual(contents[0]['inline_data']['data'], b'audio-bytes')
    self.assertFalse(os.path.exists(cmd[-1]))

  def test_removes_temp_file_if_model_call_fails(self):
    lm = mock.MagicMock()
    lm.generate_thought_and_object.side_effect = RuntimeError('LM failed')

    result = self._evaluate(lm)

    self.assertEqual(result['status'], 'error')
    self.assertLen(self.ffmpeg_cmds, 1)
    self.assertFalse(os.path.exists(self.ffmpeg_cmds[0][-1]))


if __name__ == '__main__':
  absltest.main()
