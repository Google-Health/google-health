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
"""Unit tests for post_processing module."""

from absl.testing import absltest
from absl.testing import parameterized

from data_analysis import post_processing


class PostProcessingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Children under 2 years:
      # Cutoffs: Normal >= 10.5, Mild >= 9.5, Moderate >= 7.0, Severe < 7.0
      (
          'under_2_normal',
          11.0,
          1.5,
          'Female',
          'NOT_PREGNANT',
          ('normal', 'Under 2 years'),
      ),
      (
          'under_2_mild',
          10.0,
          1.5,
          'Female',
          'NOT_PREGNANT',
          ('mild', 'Under 2 years'),
      ),
      (
          'under_2_moderate',
          8.0,
          1.5,
          'Female',
          'NOT_PREGNANT',
          ('moderate', 'Under 2 years'),
      ),
      (
          'under_2_severe',
          6.0,
          1.5,
          'Female',
          'NOT_PREGNANT',
          ('severe', 'Under 2 years'),
      ),
      # Pregnant women — Cutoffs: (11, 10, 7)
      (
          'pregnant_normal',
          11.5,
          25.0,
          'Female',
          'PREGNANT',
          ('normal', 'Pregnant'),
      ),
      ('pregnant_mild', 10.5, 25.0, 'Female', 'PREGNANT', ('mild', 'Pregnant')),
      (
          'pregnant_moderate',
          9.0,
          25.0,
          'Female',
          'PREGNANT',
          ('moderate', 'Pregnant'),
      ),
      (
          'pregnant_severe',
          6.5,
          25.0,
          'Female',
          'PREGNANT',
          ('severe', 'Pregnant'),
      ),
      # Adult males — Cutoffs: (13, 11, 8)
      (
          'adult_male_normal',
          13.5,
          30.0,
          'Male',
          'NOT_PREGNANT',
          ('normal', 'Men (>=15 years)'),
      ),
      (
          'adult_male_mild',
          12.0,
          30.0,
          'Male',
          'NOT_PREGNANT',
          ('mild', 'Men (>=15 years)'),
      ),
      (
          'adult_male_moderate',
          10.0,
          30.0,
          'Male',
          'NOT_PREGNANT',
          ('moderate', 'Men (>=15 years)'),
      ),
      (
          'adult_male_severe',
          7.5,
          30.0,
          'Male',
          'NOT_PREGNANT',
          ('severe', 'Men (>=15 years)'),
      ),
      # Adult females (not pregnant) — Cutoffs: (12, 11, 8)
      (
          'adult_female_normal',
          12.5,
          28.0,
          'Female',
          'NOT_PREGNANT',
          ('normal', 'Women (Not pregnant) (>=12 years)'),
      ),
      (
          'adult_female_mild',
          11.5,
          28.0,
          'Female',
          'NOT_PREGNANT',
          ('mild', 'Women (Not pregnant) (>=12 years)'),
      ),
  )
  def test_classify_anemia_by_hgb(
      self, hgb, age, gender, pregnancy_status, expected
  ):
    self.assertEqual(
        post_processing.classify_anemia_by_hgb(
            hgb, age, gender, pregnancy_status
        ),
        expected,
    )


if __name__ == '__main__':
  absltest.main()
