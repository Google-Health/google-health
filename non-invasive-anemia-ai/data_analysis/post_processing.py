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
# 3. Neither the name of Google Inc. nor the names of its
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
"""Post Processing utilities for Non-Invasive Anemia AI."""

from typing import Tuple


def classify_anemia_by_hgb(
    hgb: float,
    age: int,
    gender: str,
    pregnancy_status: str,
) -> Tuple[str, str]:
  """Determines anemia severity class from hemoglobin value and patient metadata.

  Cutoffs based on World Health Organization (WHO) guidelines:
  https://iris.who.int/bitstream/handle/10665/376196/9789240088542-eng.pdf

  Args:
    hgb: Continuous predicted or measured hemoglobin value in g/dL.
    age: Age of the patient in years.
    gender: Gender of the patient.
    pregnancy_status: Pregnancy status of the patient.

  Returns:
    A tuple (anemic_status, category_name) where anemic_status is one of
    'normal', 'mild', 'moderate', or 'severe'.
  """

  if age < 2:
    category = 'Under 2 years'
    cutoffs = (10.5, 9.5, 7)
  elif age < 5:
    category = '2-4 years'
    cutoffs = (11, 10, 7)
  elif age < 12:
    category = '5-11 years'
    cutoffs = (11.5, 11, 7)
  elif str(pregnancy_status).upper() == 'PREGNANT':
    category = 'Pregnant'
    # We don't have trimester, so use 1st/3rd values
    cutoffs = (11, 10, 7)
  elif str(gender).capitalize() == 'Male':
    if age < 15:
      category = 'Boys 12-14 years'
      cutoffs = (12, 11, 8)
    else:
      # guidelines only go to 65
      category = 'Men (>=15 years)'
      cutoffs = (13, 11, 8)
  else:  # all non male categories
    category = 'Women (Not pregnant) (>=12 years)'
    cutoffs = (12, 11, 8)

  if hgb >= cutoffs[0]:
    anemic = 'normal'
  elif hgb >= cutoffs[1]:
    anemic = 'mild'
  elif hgb >= cutoffs[2]:
    anemic = 'moderate'
  else:
    anemic = 'severe'
  return anemic, category
