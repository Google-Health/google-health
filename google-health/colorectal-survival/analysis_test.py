# Copyright (c) 2021, Google Inc.
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
"""Tests for analysis.py."""

import unittest
import pandas as pd

from colorectal_survival import analysis


class AnalysisTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.example_ids = [0, 1, 2, 3, 4]
    self.times = [1, 1, 2, 3, 4]  # Two examples with same time
    self.observed = [1, 1, 0, 1, 1]  # One censored example
    self.risk_scores = [7, 8, 7, 9, 10]  # One out of order, one tied
    self.df = pd.DataFrame(
        [self.example_ids, self.times, self.observed, self.risk_scores]).T
    self.df.columns = [
        'id', analysis.TIME, analysis.OBSERVED, analysis.RISK_SCORE
    ]

  def test_plot_km_curve(self):
    analysis.plot_km_curve(self.df, self.df)

  def test_discretize(self):
    risk_scores_tune = range(100)
    risk_scores_test = range(25, 125)
    expected = ['Medium Risk'] * 50 + ['High Risk'] * 50
    actual = analysis.discretize(risk_scores_tune, risk_scores_test)
    self.assertListEqual(list(actual), expected)

  def test_c_index(self):
    # Comparable pairs:
    # (0, 2): Tied
    # (0, 3): True
    # (0, 4): True
    # (1, 2): False
    # (1, 3): True
    # (1, 4): True
    # (3, 4): True
    expected = (5 + 0.5) / 7
    actual = analysis.c_index(self.df)
    self.assertEqual(expected, actual)

  def test_survival_auc(self):
    # Threshold: 1.5 Comparable pairs:
    # (0, 2): Tied
    # (0, 3): True
    # (0, 4): True
    # (1, 2): False
    # (1, 3): True
    # (1, 4): True
    expected = (4 + 0.5) / 6
    actual = analysis.survival_auc(self.df, 1.5)
    self.assertEqual(expected, actual)

  def test_get_hazard_ratios(self):
    analysis.get_hazard_ratios(
        self.df[[analysis.TIME, analysis.OBSERVED, analysis.RISK_SCORE]])


if __name__ == '__main__':
  unittest.main()
