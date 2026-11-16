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
"""Constraint utilities for structured LLM output parsing."""

import dataclasses
import json
import re
import typing as typ

import pydantic


class DataclassInstance(typ.Protocol):
  __dataclass_fields__: typ.ClassVar[dict[str, dataclasses.Field[typ.Any]]]

  def __init__(self, **kwargs: typ.Any):
    ...


T = typ.TypeVar('T', bound=DataclassInstance)


class ResponseParsingError(Exception):
  pass


@typ.runtime_checkable
@pydantic.dataclasses.dataclass
class BaseModel(typ.Protocol):

  def to_dict(self) -> dict[str, typ.Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(
      cls: typ.Type['BaseModel'], data: dict[str, typ.Any]
  ) -> 'BaseModel':
    return pydantic.TypeAdapter(cls).validate_python(data)


def json_schema(type_: typ.Type[typ.Any]) -> dict[str, typ.Any]:
  return pydantic.TypeAdapter(type_).json_schema()


def get_template(type_: typ.Type[typ.Any]) -> str:
  return json.dumps(json_schema(type_), indent=2)


def get_constraint_suffix(cls: typ.Type[typ.Any]) -> str:
  template = get_template(cls)
  return f"""
=== Answer format ===
Answer by outputting a json string (in a quoted block) based on the following schema:
{template}
"""


def parse_object(
    response: str, cls: typ.Type[T], serializer: str = 'json'
) -> T:
  """Parses a JSON object from an LLM response string.

  The JSON is read from the first ```json fenced block (case-insensitive), else
  from the first fenced block of any kind (e.g. a bare ```), else from the
  whole response.

  Args:
    response: The model response text.
    cls: The pydantic dataclass to validate the JSON object against.
    serializer: Serialization format. Only 'json' is supported.

  Returns:
    The parsed `cls` object.

  Raises:
    ValueError: If `serializer` is not 'json'.
    ResponseParsingError: If the response cannot be parsed into `cls`.
  """
  if serializer != 'json':
    raise ValueError('Only JSON serializer is supported')
  match = re.search(
      r'```json\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE
  ) or re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
  if match:
    json_str = match.group(1)
  else:
    json_str = response
  try:
    data = json.loads(json_str)
    return pydantic.TypeAdapter(cls).validate_python(data)
  except Exception as e:
    raise ResponseParsingError(f'Failed to parse response: {e}') from e
