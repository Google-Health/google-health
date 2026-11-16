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

"""Pydantic data models for ClinicalSense audio-visual evaluation data.

Each case corresponds to one simulated telehealth consultation: a video, its
transcript, and the clinician-authored annotations describing the audio-visual
findings the case is meant to test.
"""

import typing as typ
import pydantic
from pydantic.dataclasses import dataclass
import constraints


@dataclass
class AudioVideoIndicator(constraints.BaseModel):
  av_indicator: str
  is_continuous: bool
  time_windows: typ.Optional[list[tuple[float, float]]]


@dataclass
class AudioVideoIndicatorList(constraints.BaseModel):
  av_indicators: list[AudioVideoIndicator]


@dataclass
class TranscriptTurn(constraints.BaseModel):
  speaker: typ.Literal['Doctor', 'Patient']
  text: str
  start_time: float
  end_time: float
  medically_significant: bool

  def as_markdown(self) -> str:
    return (
        f'**{self.speaker}** [{self.start_time:.2f} - {self.end_time:.2f}]:'
        f' {self.text}'
    )


@dataclass
class Transcript(constraints.BaseModel):
  turns: list[TranscriptTurn]


@dataclass
class AvEvaluationCase(constraints.BaseModel):
  case_id: str
  case_condition: str
  file_name: str
  key_medical_information: str
  audio_video_indicators: AudioVideoIndicatorList
  raw_video_path: str
  raw_transcript_path: str
  transcript: Transcript
  summary_keywords: list[str] = pydantic.Field(default_factory=list)
  patient_sex: typ.Optional[typ.Literal['Male', 'Female']] = None
