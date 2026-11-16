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
"""Unit tests for Gemini media part helpers."""

import types
from unittest import mock

from absl.testing import absltest

import content_utils


class MediaChunkTest(absltest.TestCase):

  def test_video_chunk_is_inline_data_part(self):
    path = self.create_tempfile('clip.mp4', content='video-bytes').full_path
    part = content_utils.video_chunk(path)
    self.assertEqual(
        part,
        {'inline_data': {'mime_type': 'video/mp4', 'data': b'video-bytes'}},
    )

  def test_video_chunk_with_fps_sets_video_metadata(self):
    path = self.create_tempfile('clip.mp4', content='video-bytes').full_path
    part = content_utils.video_chunk(path, fps=5.0)
    self.assertEqual(part['video_metadata'], {'fps': 5.0})
    self.assertEqual(part['inline_data']['mime_type'], 'video/mp4')

  def test_audio_chunk_normalizes_wav_mimetype(self):
    path = self.create_tempfile('clip.wav', content='audio-bytes').full_path
    part = content_utils.audio_chunk(path)
    self.assertEqual(part['inline_data']['mime_type'], 'audio/wav')
    self.assertEqual(part['inline_data']['data'], b'audio-bytes')
    self.assertNotIn('video_metadata', part)

  def test_explicit_mimetype_wins(self):
    path = self.create_tempfile('clip.bin', content='x').full_path
    part = content_utils.video_chunk(path, mimetype='video/webm')
    self.assertEqual(part['inline_data']['mime_type'], 'video/webm')


class UploadFileTest(absltest.TestCase):

  def _file(self, state):
    return types.SimpleNamespace(
        name='files/abc',
        uri='https://example.com/files/abc',
        mime_type='video/mp4',
        state=state,
    )

  @mock.patch.object(content_utils.time, 'sleep', autospec=True)
  def test_waits_until_active(self, mock_sleep):
    client = mock.MagicMock()
    client.files.upload.return_value = self._file('PROCESSING')
    client.files.get.side_effect = [
        self._file('PROCESSING'),
        self._file('ACTIVE'),
    ]

    uploaded = content_utils.upload_file(client, '/tmp/video.mp4')

    self.assertEqual(uploaded.state, 'ACTIVE')
    client.files.upload.assert_called_once_with(
        file='/tmp/video.mp4', config={'mime_type': 'video/mp4'}
    )
    self.assertEqual(client.files.get.call_count, 2)
    self.assertEqual(mock_sleep.call_count, 2)

  def test_failed_processing_raises(self):
    client = mock.MagicMock()
    client.files.upload.return_value = self._file('FAILED')
    with self.assertRaisesRegex(RuntimeError, 'Processing failed'):
      content_utils.upload_file(client, '/tmp/video.mp4')

  def test_file_chunk_references_uploaded_file(self):
    part = content_utils.file_chunk(self._file('ACTIVE'), fps=10.0)
    self.assertEqual(
        part,
        {
            'file_data': {
                'file_uri': 'https://example.com/files/abc',
                'mime_type': 'video/mp4',
            },
            'video_metadata': {'fps': 10.0},
        },
    )


class MediaPartTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.client = mock.MagicMock()
    self.client.files.upload.return_value = types.SimpleNamespace(
        name='files/abc',
        uri='https://example.com/files/abc',
        mime_type='video/mp4',
        state='ACTIVE',
    )
    # 11 bytes are "large" for these tests.
    self.enter_context(mock.patch.object(content_utils, 'MAX_INLINE_BYTES', 10))

  def test_small_video_is_sent_inline(self):
    path = self.create_tempfile('clip.mp4', content='small').full_path
    with content_utils.media_part(self.client, path, fps=5.0) as part:
      self.assertEqual(part['inline_data']['data'], b'small')
      self.assertEqual(part['video_metadata'], {'fps': 5.0})
    self.client.files.upload.assert_not_called()

  def test_small_audio_is_sent_as_audio(self):
    path = self.create_tempfile('clip.wav', content='small').full_path
    with content_utils.media_part(self.client, path) as part:
      self.assertEqual(part['inline_data']['mime_type'], 'audio/wav')
      self.assertNotIn('video_metadata', part)

  def test_large_video_is_uploaded_and_deleted(self):
    path = self.create_tempfile('clip.mp4', content='video-bytes').full_path
    with content_utils.media_part(self.client, path, fps=5.0) as part:
      self.assertEqual(
          part['file_data']['file_uri'], 'https://example.com/files/abc'
      )
      self.assertEqual(part['video_metadata'], {'fps': 5.0})
      self.client.files.delete.assert_not_called()
    self.client.files.delete.assert_called_once_with(name='files/abc')

  def test_uploaded_file_is_deleted_if_the_request_fails(self):
    path = self.create_tempfile('clip.mp4', content='video-bytes').full_path
    with self.assertRaisesRegex(RuntimeError, 'request failed'):
      with content_utils.media_part(self.client, path):
        raise RuntimeError('request failed')
    self.client.files.delete.assert_called_once_with(name='files/abc')

  def test_large_file_is_sent_inline_without_a_client(self):
    path = self.create_tempfile('clip.mp4', content='video-bytes').full_path
    with content_utils.media_part(None, path) as part:
      self.assertEqual(part['inline_data']['data'], b'video-bytes')


if __name__ == '__main__':
  absltest.main()
