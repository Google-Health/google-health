# Copyright 2026 Google LLC. All rights reserved.
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

"""Helpers for building media parts for the Gemini API.

The google-genai SDK takes a list of "parts" as its `contents` argument, where
each part is either a plain string (for text) or a part dict (for media), e.g.
{'inline_data': {'mime_type': ..., 'data': ...}}. These helpers turn a local
media file into such a part dict, so callers can build request contents as an
ordinary list, e.g.:

    contents = [content_utils.video_chunk(clip_path, fps=5.0), prompt]

Inline data is capped at ~20MB per request (after base64 encoding). For larger
files, upload them with `upload_file` and pass `file_chunk(uploaded)` instead,
or use `media_part`, which does either depending on the file size:

    with content_utils.media_part(lm.client, clip_path, fps=5.0) as part:
      result = lm.generate_thought_and_object([part, prompt], cls=...)
"""

import contextlib
import mimetypes
import os
import time
from typing import Any, Iterator

from absl import logging

_DEFAULT_VIDEO_MIMETYPE = 'video/mp4'
_DEFAULT_AUDIO_MIMETYPE = 'audio/mpeg'
# Python's mimetypes table uses legacy names that Gemini does not list.
_MIMETYPE_ALIASES = {'audio/x-wav': 'audio/wav'}

# Media files larger than this should be sent through the Files API instead of
# inline. Inline data is base64-encoded (+33%), which leaves headroom below the
# ~20MB request limit.
MAX_INLINE_BYTES = 14 * 1024 * 1024


def _guess_mimetype(path: str, mimetype: str | None) -> str | None:
  """Returns mimetype, or the type guessed from the extension if it is None."""
  if mimetype is None:
    mimetype, _ = mimetypes.guess_type(path)
  return _MIMETYPE_ALIASES.get(mimetype or '', mimetype)


def _inline_data(
    path: str, mimetype: str | None, default_mimetype: str
) -> dict[str, Any]:
  """Reads a local file and returns it as a Gemini inline-data blob."""
  mimetype = _guess_mimetype(path, mimetype)
  with open(path, 'rb') as f:
    return {'mime_type': mimetype or default_mimetype, 'data': f.read()}


def _with_video_metadata(
    part: dict[str, Any], fps: float | None
) -> dict[str, Any]:
  """Adds the video sampling rate to a part, if one is given."""
  if fps is not None:
    part['video_metadata'] = {'fps': fps}
  return part


def video_chunk(
    video_path: str, mimetype: str | None = None, fps: float | None = None
) -> dict[str, Any]:
  """Returns a local video file as an inline-data part.

  Args:
    video_path: Path to a local video file.
    mimetype: MIME type of the file. Guessed from the extension if None.
    fps: Frame rate at which Gemini samples the video, in (0, 24]. If None,
      Gemini uses its default (1 FPS).

  Returns:
    A part dict accepted by the google-genai SDK.
  """
  part = {
      'inline_data': _inline_data(video_path, mimetype, _DEFAULT_VIDEO_MIMETYPE)
  }
  return _with_video_metadata(part, fps)


def audio_chunk(audio_path: str, mimetype: str | None = None) -> dict[str, Any]:
  """Returns a local audio file as an inline-data part."""
  return {
      'inline_data': _inline_data(audio_path, mimetype, _DEFAULT_AUDIO_MIMETYPE)
  }


def _file_state(uploaded_file: Any) -> str:
  """Returns the processing state of an uploaded file, e.g. 'ACTIVE'."""
  state = getattr(uploaded_file, 'state', None)
  return str(getattr(state, 'value', state) or '').upper()


def upload_file(
    client: Any,
    path: str,
    mimetype: str | None = None,
    poll_interval_sec: float = 2.0,
    timeout_sec: float = 600.0,
) -> Any:
  """Uploads a local file with the Gemini Files API and waits until it is ready.

  Uploaded files are kept by the service for a limited time; callers that
  upload many files should delete them with `client.files.delete(name=...)`
  once they are no longer needed.

  Args:
    client: A `google.genai.Client`.
    path: Path to the local file.
    mimetype: MIME type of the file. Guessed from the extension if None.
    poll_interval_sec: Seconds between processing-state checks.
    timeout_sec: Maximum number of seconds to wait for processing.

  Returns:
    The uploaded `google.genai.types.File`, in the ACTIVE state.

  Raises:
    TimeoutError: If the file is still being processed after `timeout_sec`.
    RuntimeError: If the service failed to process the file.
  """
  mimetype = _guess_mimetype(path, mimetype)
  config = {'mime_type': mimetype} if mimetype else None
  uploaded = client.files.upload(file=path, config=config)
  deadline = time.monotonic() + timeout_sec
  while _file_state(uploaded) == 'PROCESSING':
    if time.monotonic() > deadline:
      raise TimeoutError(
          f'Timed out waiting for uploaded file {uploaded.name} ({path}).'
      )
    time.sleep(poll_interval_sec)
    uploaded = client.files.get(name=uploaded.name)
  if _file_state(uploaded) == 'FAILED':
    raise RuntimeError(f'Processing failed for uploaded file {path}.')
  return uploaded


def file_chunk(uploaded_file: Any, fps: float | None = None) -> dict[str, Any]:
  """Returns a part that references a file uploaded with `upload_file`.

  Args:
    uploaded_file: A `google.genai.types.File` returned by `upload_file`.
    fps: For videos, the frame rate at which Gemini samples the video, in (0,
      24]. If None, Gemini uses its default (1 FPS).

  Returns:
    A part dict accepted by the google-genai SDK.
  """
  part = {
      'file_data': {
          'file_uri': uploaded_file.uri,
          'mime_type': uploaded_file.mime_type,
      }
  }
  return _with_video_metadata(part, fps)


@contextlib.contextmanager
def media_part(
    client: Any,
    path: str,
    mimetype: str | None = None,
    fps: float | None = None,
) -> Iterator[dict[str, Any]]:
  """Yields a part for a local video or audio file, uploading it if needed.

  Files of up to MAX_INLINE_BYTES are sent inline. Larger files (e.g. long or
  high-resolution clips) are uploaded with the Files API and deleted again when
  the context exits, so a request never exceeds the inline size limit.

  Args:
    client: A `google.genai.Client` for uploads, or None to always send the file
      inline.
    path: Path to the local media file.
    mimetype: MIME type of the file. Guessed from the extension if None.
    fps: For videos, the frame rate at which Gemini samples the video, in (0,
      24]. If None, Gemini uses its default (1 FPS).

  Yields:
    A part dict accepted by the google-genai SDK.
  """
  if client is None or os.path.getsize(path) <= MAX_INLINE_BYTES:
    if (_guess_mimetype(path, mimetype) or '').startswith('audio/'):
      yield audio_chunk(path, mimetype=mimetype)
    else:
      yield video_chunk(path, mimetype=mimetype, fps=fps)
    return
  uploaded = upload_file(client, path, mimetype=mimetype)
  try:
    yield file_chunk(uploaded, fps=fps)
  finally:
    try:
      client.files.delete(name=uploaded.name)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('Could not delete uploaded file %s: %s', uploaded.name, e)
