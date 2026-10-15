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
"""Unit tests for metrics calculation module."""

from unittest import mock

from absl.testing import absltest
import matplotlib
import pandas as pd

from evaluation import metrics
from evaluation import sub_group


matplotlib.use('Agg')  # Headless backend for automated testing


class MetricsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.uuids = ['id1', 'id2', 'id3', 'id4', 'id5']
    self.metadata = pd.DataFrame(
        {
            'Age': [25.0, 30.0, 1.0, 22.0, 45.0],
            'ground_truth_hgb': [13.0, 10.0, 6.5, 11.5, 7.0],
            'ground_truth_anemia_label': [
                'normal',
                'moderate',
                'severe',
                'mild',
                'severe',
            ],
            'PregnancyStatus': [
                'NOT_PREGNANT',
                'NOT_PREGNANT',
                'NOT_PREGNANT',
                'PREGNANT',
                'NOT_PREGNANT',
            ],
            'Gender': ['Female', 'Male', 'Female', 'Female', 'Male'],
        },
        index=self.uuids,
    )

    self.y_true = self.metadata['ground_truth_hgb'].values
    self.y_pred = [12.8, 9.8, 6.2, 11.2, 7.1]
    self.eval_df = pd.DataFrame({
        'uuid': self.uuids,
        'gt_hgb': self.y_true,
        'gt_anemia_label': self.metadata['ground_truth_anemia_label'].values,
        'pred_hgb': self.y_pred,
        'pred_anemia_label': [
            'normal',
            'moderate',
            'severe',
            'mild',
            'severe',
        ],
    })
    self.feature_columns = [
        'Age',
        'ground_truth_hgb',
        'ground_truth_anemia_label',
        'PregnancyStatus',
        'Gender',
    ]

  def test_calculate_multiclass_metrics_regression(self):
    """Tests the calculation of multiclass metrics for regression."""

    multiclass_metrics = metrics.calculate_multiclass_metrics_regression(
        y_true=self.y_true,
        y_pred=self.y_pred,
        uuids=self.uuids,
        metadata=self.metadata,
        feature_columns=self.feature_columns,
    )
    self.assertIn('overall_accuracy', multiclass_metrics)
    self.assertIn('overall_accuracy_tolerance', multiclass_metrics)
    self.assertIn('class_0', multiclass_metrics)
    self.assertIn('class_1', multiclass_metrics)
    self.assertIn('class_2', multiclass_metrics)
    self.assertIn('class_3', multiclass_metrics)

  def test_calculate_multiclass_metrics_regression_with_plot(self):
    """Tests multiclass metrics calculation when plot=True."""
    with (
        mock.patch.object(
            metrics.sns, 'heatmap', wraps=metrics.sns.heatmap
        ) as mock_heatmap,
        mock.patch.object(metrics.plt, 'show') as mock_show,
    ):
      metrics.calculate_multiclass_metrics_regression(
          y_true=self.y_true,
          y_pred=self.y_pred,
          uuids=self.uuids,
          metadata=self.metadata,
          feature_columns=self.feature_columns,
          plot=True,
      )
    # 1. Verify plotting functions were called
    mock_heatmap.assert_called_once()
    mock_show.assert_called_once()
    # 2. Verify heatmap arguments (confusion matrix shape, labels, colormap)
    heatmap_args, heatmap_kwargs = mock_heatmap.call_args
    cm_arg = heatmap_args[0]
    self.assertEqual(cm_arg.shape, (4, 4))
    self.assertEqual(heatmap_kwargs['cmap'], 'Blues')
    self.assertEqual(
        heatmap_kwargs['xticklabels'],
        ['Normal', 'Mild', 'Moderate', 'Severe'],
    )

  def test_sub_group_metrics_regression(self):
    """Tests the calculation of subgroup metrics for regression."""

    metrics_per_class, metrics_per_binary_class = (
        sub_group.sub_group_metrics_regression(
            eval_df=self.eval_df,
            metadata=self.metadata,
            feature_columns=self.feature_columns,
        )
    )
    self.assertIn('overall_mae', metrics_per_class)
    self.assertIn('class_0', metrics_per_class)
    self.assertIn('class_1', metrics_per_class)
    self.assertIn('class_2', metrics_per_class)
    self.assertIn('class_3', metrics_per_class)
    self.assertIn('anemia_vs_no_anemia', metrics_per_binary_class)
    self.assertIn(
        'sensitivity', metrics_per_binary_class['anemia_vs_no_anemia']
    )

  def test_map_regression_results_to_binary_classification(self):
    """Tests the mapping of regression results to binary classification."""

    (
        pred_df,
        mae,
        sensitivity,
        specificity,
        _,
        _,
        _,
        _,
        _,
        sens_ci,
        spec_ci,
    ) = metrics.map_regression_results_to_binary_classification(
        class_map_tag='anemia_vs_no_anemia',
        metadata=self.metadata,
        feature_columns=self.feature_columns,
        ground_truth_hgb=self.y_true,
        predicted_hgb=self.y_pred,
        patient_uuids=self.uuids,
    )

    self.assertIsInstance(pred_df, pd.DataFrame)
    self.assertLen(pred_df, 5)
    self.assertAlmostEqual(mae, 0.22, places=2)
    self.assertGreaterEqual(sensitivity, 0.0)
    self.assertLessEqual(sensitivity, 1.0)
    self.assertGreaterEqual(specificity, 0.0)
    self.assertLessEqual(specificity, 1.0)
    self.assertIsInstance(sens_ci, str)
    self.assertIsInstance(spec_ci, str)


if __name__ == '__main__':
  absltest.main()
