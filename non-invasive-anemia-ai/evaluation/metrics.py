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

"""Metrics calculation utility functions."""

import enum
import functools
import statistics
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from data_analysis.post_processing import classify_anemia_by_hgb


class AnemiaClass(enum.IntEnum):
  """Enumeration for the 4 standard anemia severity classes."""

  NORMAL = 0
  MILD = 1
  MODERATE = 2
  SEVERE = 3


# Mapping of 4 anemia classes to binary classes.
# 0: normal, 1: mild, 2: moderate, 3: severe
anemia_vs_no_anemia_map = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
}

severe_vs_not_severe_anemia_map = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
}

moderate_severe_vs_normal_mild_anemia_map = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
}

"""Defines binary label mappings for each anemia classification task by grouping
the original four anemia severity classes into negative (0) and positive (1).
"""
ANEMIA_CLASSIFICATION_BINARY_LABEL_MAPS = {
    'anemia_vs_no_anemia': anemia_vs_no_anemia_map,
    'moderate_severe_vs_normal_mild_anemia': (
        moderate_severe_vs_normal_mild_anemia_map
    ),
    'severe_vs_not_severe_anemia': severe_vs_not_severe_anemia_map,
}


def compute_binary_classification_metrics(
    anemia_pos_true: np.ndarray,
    anemia_pos_pred: np.ndarray,
    anemia_neg_true: np.ndarray,
    anemia_neg_pred: np.ndarray,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    tuple[float, float],
    tuple[float, float],
]:
  """Computes binary classification metrics.

  Args:
    anemia_pos_true: True positive labels.
    anemia_pos_pred: Predicted positive labels.
    anemia_neg_true: True negative labels.
    anemia_neg_pred: Predicted negative labels.

  Returns:
    A tuple containing sensitivity, specificity, ppv, npv, sens_spec_hm,
    sens_ci, and spec_ci.
  """
  sensitivity = float(
      recall_score(anemia_pos_true, anemia_pos_pred, zero_division=0)
  )
  specificity = float(
      recall_score(anemia_neg_true, anemia_neg_pred, zero_division=0)
  )
  ppv = float(
      precision_score(anemia_pos_true, anemia_pos_pred, zero_division=0)
  )
  npv = float(
      precision_score(anemia_neg_true, anemia_neg_pred, zero_division=0)
  )

  if sensitivity + specificity > 0:
    sens_spec_hm = float(statistics.harmonic_mean([sensitivity, specificity]))
  else:
    sens_spec_hm = 0.0

  try:
    recall_fn = functools.partial(recall_score, zero_division=0)
    bootstrap_pos = stats.bootstrap(
        (anemia_pos_true, anemia_pos_pred),
        recall_fn,
        n_resamples=1000,
        paired=True,
        random_state=0,
    )
    sens_ci = (
        bootstrap_pos.confidence_interval.low,
        bootstrap_pos.confidence_interval.high,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    sens_ci = (sensitivity, sensitivity)

  try:
    recall_fn = functools.partial(recall_score, zero_division=0)
    bootstrap_neg = stats.bootstrap(
        (anemia_neg_true, anemia_neg_pred),
        recall_fn,
        n_resamples=1000,
        paired=True,
        random_state=0,
    )
    spec_ci = (
        bootstrap_neg.confidence_interval.low,
        bootstrap_neg.confidence_interval.high,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    spec_ci = (specificity, specificity)
  return (
      sensitivity,
      specificity,
      ppv,
      npv,
      sens_spec_hm,
      sens_ci,
      spec_ci,
  )


def map_regression_results_to_binary_classification(
    class_map_tag: str,
    metadata: pd.DataFrame,
    feature_columns: List[str],
    ground_truth_hgb: float,
    predicted_hgb: float,
    patient_uuids: Any,
    pos_label: int = 1,
    neg_label: int = 0,
) -> Tuple[
    pd.DataFrame,
    float,
    float,
    float,
    float,
    float,
    float,
    pd.DataFrame,
    pd.DataFrame,
    str,
    str,
]:
  """Maps regression results to binary classification and computes metrics.

  Args:
    class_map_tag: Mapping key in ANEMIA_CLASSIFICATION_BINARY_LABEL_MAPS.
    metadata: Dataset Metadata DataFrame indexed by patient/sample UUIDs.
    feature_columns: Feature columns to select from metadata.
      Expected:
          [
              'Age',
              'ground_truth_hgb',
              'ground_truth_anemia_label',
              'PregnancyStatus',
              'Gender',
          ]
      Note:
          Feature columns should be a subset of the metadata columns.
    ground_truth_hgb: Ground truth hemoglobin value in g/dL.
    predicted_hgb: Predicted hemoglobin value in g/dL.
    patient_uuids: Patient/sample UUIDs.
    pos_label: Binary positive label value (default: 1).
    neg_label: Binary negative label value (default: 0).

  Returns:
    A tuple containing (predictions_df, mae, sensitivity, specificity,
    sens_spec_hm, ppv, npv, severe_caught_df, severe_missed_df, sens_ci_str,
    spec_ci_str).
  """
  class_mapper = ANEMIA_CLASSIFICATION_BINARY_LABEL_MAPS[class_map_tag]

  if metadata is None:
    raise ValueError('metadata DataFrame must be provided.')
  if feature_columns is None:
    raise ValueError('feature columns must be provided.')

  predictions_df = (
      metadata.loc[patient_uuids][feature_columns].copy().reset_index()
  )

  predictions_df['y_true'] = ground_truth_hgb
  predictions_df['y_pred'] = predicted_hgb
  predictions_df['predicted_class'] = predictions_df.apply(
      lambda row: classify_anemia_by_hgb(
          row.y_pred, row.Age, row.Gender, row.PregnancyStatus
      )[0],
      axis=1,
  )

  def _get_class_index(label_str: str) -> int:
    try:
      return AnemiaClass[label_str.upper()].value
    except (KeyError, AttributeError):
      return 0

  predictions_df['mapped_label'] = predictions_df.apply(
      lambda row: class_mapper[
          _get_class_index(str(row.ground_truth_anemia_label))
      ],
      axis=1,
  )
  predictions_df['mapped_predicted_class'] = predictions_df.apply(
      lambda row: class_mapper[_get_class_index(str(row.predicted_class))],
      axis=1,
  )

  # Check how many severe cases were captured vs missed
  severe_mask = (
      predictions_df['ground_truth_anemia_label'].str.lower() == 'severe'
  )
  severe_caught_df = predictions_df[
      severe_mask & (predictions_df['mapped_predicted_class'] == 1)
  ]
  severe_missed_df = predictions_df[
      severe_mask & (predictions_df['mapped_predicted_class'] == 0)
  ]

  anemia_pos_true = np.array(predictions_df['mapped_label']) == pos_label
  anemia_pos_pred = (
      np.array(predictions_df['mapped_predicted_class']) == pos_label
  )
  anemia_neg_true = np.array(predictions_df['mapped_label']) == neg_label
  anemia_neg_pred = (
      np.array(predictions_df['mapped_predicted_class']) == neg_label
  )

  (
      sensitivity,
      specificity,
      ppv,
      npv,
      sens_spec_hm,
      sens_ci,
      spec_ci,
  ) = compute_binary_classification_metrics(
      anemia_pos_true,
      anemia_pos_pred,
      anemia_neg_true,
      anemia_neg_pred,
  )

  mae = float(mean_absolute_error(ground_truth_hgb, predicted_hgb))

  return (
      predictions_df,
      mae,
      sensitivity,
      specificity,
      sens_spec_hm,
      ppv,
      npv,
      severe_caught_df,
      severe_missed_df,
      f'{sens_ci[0]:.2f}-{sens_ci[1]:.2f}',
      f'{spec_ci[0]:.2f}-{spec_ci[1]:.2f}',
  )


def compute_multiclass_metrics(
    cm: np.ndarray,
    predictions_df: pd.DataFrame,
    tolerance: float,
    num_classes: int,
) -> Dict[str, Any]:
  """Computes metrics from the confusion matrix and predictions DataFrame.

  Args:
    cm: Confusion matrix as a numpy array.
    predictions_df: Predictions DataFrame.
    tolerance: Error tolerance limit in g/dL for tolerance accuracy.
    num_classes: Number of discrete ordinal classes.

  Returns:
    Dictionary containing overall and per-class classification and regression
    metrics.
  """
  correct_predictions = float(np.sum(cm.diagonal()))
  total_predictions = float(np.sum(cm))
  overall_accuracy = (
      correct_predictions / total_predictions if total_predictions > 0 else 0.0
  )

  absolute_difference = np.abs(
      predictions_df['y_true'] - predictions_df['y_pred']
  )
  correct_tol = float(np.sum(absolute_difference <= tolerance))
  total_samples_cnt = float(len(predictions_df['y_true']))
  overall_accuracy_tolerance = (
      correct_tol / total_samples_cnt if total_samples_cnt > 0 else 0.0
  )
  overall_percent_tolerance = overall_accuracy_tolerance

  metrics: Dict[str, Any] = {
      'overall_percent_tolerance': overall_percent_tolerance,
      'overall_accuracy_tolerance': overall_accuracy_tolerance,
      'overall_accuracy': overall_accuracy,
  }

  # Compute per-class metrics
  for i in range(num_classes):
    tp = float(cm[i, i])
    fn = float(np.sum(cm[i, :]) - tp)
    fp = float(np.sum(cm[:, i]) - tp)
    tn = float(np.sum(cm) - (tp + fp + fn))

    predictions_current_class = predictions_df[
        predictions_df['mapped_label'] == i
    ]
    abs_diff_cls = np.abs(
        predictions_current_class['y_true']
        - predictions_current_class['y_pred']
    )
    total_cls = float(len(predictions_current_class))
    corr_cls = float(np.sum(abs_diff_cls <= tolerance))
    accuracy_tol = corr_cls / total_cls if total_cls > 0 else 0.0

    total_samples = tp + tn + fp + fn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (
        2 * (precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    if total_cls > 0:
      mae = float(
          np.mean(
              np.abs(
                  predictions_current_class['y_pred']
                  - predictions_current_class['y_true']
              )
          )
      )
    else:
      mae = 0.0

    metrics[f'class_{i}'] = {
        'accuracy_tolerance': accuracy_tol,
        'accuracy': accuracy,
        'sensitivity_recall': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'npv': npv,
        'mae': mae,
    }

  return metrics


def calculate_multiclass_metrics_regression(
    y_true: Any,
    y_pred: Any,
    uuids: Any,
    metadata: pd.DataFrame,
    feature_columns: List[str],
    num_classes: int = 4,
    tolerance: float = 1.0,
    plot: bool = False,
) -> Dict[str, Any]:
  """Calculates multiclass confusion matrix and regression metrics.

  Args:
    y_true: True hemoglobin values array-like.
    y_pred: Predicted hemoglobin values array-like.
    uuids: List of sample/patient identifiers.
    metadata: Demographic metadata DataFrame indexed by UUIDs.
    feature_columns: Feature columns to select from metadata.
      Expected:
          [
              'Age',
              'ground_truth_hgb',
              'ground_truth_anemia_label',
              'PregnancyStatus',
              'Gender',
          ]
      Note:
          Feature columns should be a subset of the metadata columns.
    num_classes: Number of discrete ordinal classes (default: 4).
    tolerance: Error tolerance limit in g/dL for tolerance accuracy.
    plot: Whether to plot the confusion matrix.

  Returns:
    Dictionary containing overall and per-class classification and regression
    metrics.
  """
  if metadata is None:
    raise ValueError('metadata DataFrame must be provided.')

  predictions_df = metadata.loc[uuids][feature_columns].copy().reset_index()
  predictions_df['y_true'] = np.array(y_true)
  predictions_df['y_pred'] = np.array(y_pred)

  def _get_class_index(label_str: str) -> int:
    try:
      return AnemiaClass[str(label_str).upper()].value
    except (KeyError, AttributeError):
      return 0

  predictions_df['predicted_class'] = predictions_df.apply(
      lambda row: classify_anemia_by_hgb(
          row['y_pred'], row['Age'], row['Gender'], row['PregnancyStatus']
      )[0],
      axis=1,
  )
  predictions_df['mapped_predicted_class'] = predictions_df.apply(
      lambda row: _get_class_index(str(row['predicted_class'])), axis=1
  )

  predictions_df['mapped_label'] = predictions_df.apply(
      lambda row: _get_class_index(str(row['ground_truth_anemia_label'])),
      axis=1,
  )

  all_classes = list(range(num_classes))
  cm = confusion_matrix(
      predictions_df['mapped_label'],
      predictions_df['mapped_predicted_class'],
      labels=all_classes,
  )
  with np.errstate(all='ignore'):
    cm_perc = np.nan_to_num(
        confusion_matrix(
            predictions_df['mapped_label'],
            predictions_df['mapped_predicted_class'],
            normalize='true',
            labels=all_classes,
        )
    )

  class_labels = ['Normal', 'Mild', 'Moderate', 'Severe'][:num_classes]
  counts = cm.flatten()
  percentages = cm_perc.flatten()
  labels = [f'{v1}\n({v2:.2%})' for v1, v2 in zip(counts, percentages)]
  labels_arr = np.asarray(labels).reshape(cm.shape)

  if plot:
    _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels_arr,
        fmt='',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.show()

  return compute_multiclass_metrics(cm, predictions_df, tolerance, num_classes)
