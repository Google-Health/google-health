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
"""Unit tests for the plotting module."""

from absl.testing import absltest

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from plotting import statistical

matplotlib.use('Agg')  # Headless backend for automated testing


class StatisticalPlottingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.uuids = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6']
    self.eval_df = pd.DataFrame(
        {
            'uuid': self.uuids,
            'gt_hgb': [13.5, 11.2, 9.5, 6.8, 14.0, 10.2],
            'pred_hgb': [13.2, 11.0, 9.8, 7.1, 13.8, 10.0],
            'pred_anemia_label': [
                'normal',
                'mild',
                'moderate',
                'severe',
                'normal',
                'mild',
            ],
            'ground_truth_anemia_label': [
                'normal',
                'mild',
                'moderate',
                'severe',
                'normal',
                'mild',
            ],
            'Age': [25.0, 30.0, 22.0, 45.0, 50.0, 2.5],
            'PregnancyStatus': [
                'NOT_PREGNANT',
                'NOT_PREGNANT',
                'PREGNANT',
                'NOT_PREGNANT',
                'NOT_PREGNANT',
                'NOT_PREGNANT',
            ],
            'Gender': ['Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
        },
        index=self.uuids,
    )

  def test_plot_predicted_vs_actual_hgb(self):
    """Tests the plotting of predicted vs. actual hemoglobin."""
    fig = statistical.plot_predicted_vs_actual_hgb(
        self.eval_df, show_plot=False
    )
    self.assertIsInstance(fig, plt.Figure)
    plt.close(fig)

  def test_plot_bland_altman(self):
    """Tests the plotting of Bland-Altman plot."""
    fig = statistical.plot_bland_altman(self.eval_df, show_plot=False)
    self.assertIsInstance(fig, plt.Figure)
    plt.close(fig)


if __name__ == '__main__':
  absltest.main()
