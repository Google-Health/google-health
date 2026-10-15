# Copyright (c) 2025, Google Inc.
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

"""Subgroup metrics calculation utility functions."""

from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

from evaluation.metrics import ANEMIA_CLASSIFICATION_BINARY_LABEL_MAPS
from evaluation.metrics import calculate_multiclass_metrics_regression
from evaluation.metrics import map_regression_results_to_binary_classification


warnings.filterwarnings('ignore')


def sub_group_metrics_regression(
    eval_df: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Computes demographic subgroup metrics from regression predictions.

  Args:
    eval_df: Evaluation results dataframe consisting of following columns:
        - uuid: Patient/sample identifier.
        - gt_hgb: Ground truth hemoglobin value in g/dL.
        - gt_anemia_label: Ground truth anemia label.
        - pred_hgb: Predicted hemoglobin value in g/dL.
        - pred_anemia_label: Predicted anemia label.
    metadata:  Dataset Metadata DataFrame indexed by patient/sample UUIDs.
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

  Returns:
    A tuple containing (metrics_per_class, metrics_per_binary_class).
    metrics_per_class: Dictionary containing metrics per class.
    metrics_per_binary_class: Dictionary containing binary metrics per class.
  """

  y_true, y_pred, pred_patient_ids = (
      eval_df.gt_hgb,
      eval_df.pred_hgb,
      eval_df.uuid.tolist(),
  )

  age_uuids = np.intersect1d(
      pred_patient_ids, metadata.index.tolist()
  )
  pred_patient_ids_df = pd.DataFrame(pred_patient_ids, columns=['uuid'])
  pred_age_ids = pred_patient_ids_df[
      pred_patient_ids_df['uuid'].isin(age_uuids)
  ]
  age_indices = pred_age_ids.index.tolist()

  y_true_age = y_true.iloc[age_indices].reset_index(drop=True)
  y_pred_age = y_pred.iloc[age_indices].reset_index(drop=True)

  overall_mae = float(np.mean(np.abs(y_pred_age - y_true_age)))

  metrics_per_class = calculate_multiclass_metrics_regression(
      y_true_age,
      y_pred_age,
      age_uuids,
      metadata=metadata,
      feature_columns=feature_columns,
      num_classes=4,
      tolerance=1.0,
  )
  metrics_per_class['overall_mae'] = overall_mae
  metrics_per_binary_class = {}

  for class_map_tag in ANEMIA_CLASSIFICATION_BINARY_LABEL_MAPS:
    (
        _,
        _,
        sensitivity,
        specificity,
        sens_spec_hm,
        ppv,
        npv,
        severe_caught_df,
        severe_missed_df,
        sens_ci,
        spec_ci,
    ) = map_regression_results_to_binary_classification(
        class_map_tag, metadata, feature_columns, y_true_age, y_pred_age,
        age_uuids,
    )
    n_caught_samples = len(severe_caught_df)
    n_missed_samples = len(severe_missed_df)
    n_severe_samples = n_missed_samples + n_caught_samples
    pc_severe_missed = (
        np.round(100 * n_missed_samples / n_severe_samples, 2)
        if n_severe_samples > 0
        else 0
    )
    metrics_per_binary_class[class_map_tag] = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'sens_spec_hm': sens_spec_hm,
        'ppv': ppv,
        'npv': npv,
        'sens_ci': sens_ci,
        'spec_ci': spec_ci,
        'pc_severe_missed': pc_severe_missed,
        'n_caught_samples': n_caught_samples,
        'n_missed_samples': n_missed_samples,
        'n_severe_samples': n_severe_samples,
    }

  return metrics_per_class, metrics_per_binary_class


def print_metrics(
    metrics_per_class: Dict[str, Any],
    metrics_per_binary_class: Dict[str, Any],
) -> None:
  """Prints metrics for a given subgroup.

  Args:
    metrics_per_class: Dictionary containing metrics per class.
    metrics_per_binary_class: Dictionary containing binary metrics per class.

  Returns:
    None
  """

  print(f'\nOverall Accuracy:  {metrics_per_class["overall_accuracy"]:.2f}')
  print(
      '\nclass id         Precision    Sensitivity(Recall)    Specificity    '
      ' F1 Score     Accuracy'
  )
  for class_id in metrics_per_class.keys():
    if (
        class_id != 'overall_accuracy'
        and class_id != 'overall_accuracy_tolerance'
        and class_id != 'overall_percent_tolerance'
    ):
      print(
          f'\n{class_id}         '
          f' {metrics_per_class[class_id]["precision"]:.2f}         '
          f' {metrics_per_class[class_id]["sensitivity_recall"]:.2f}           '
          f'       {metrics_per_class[class_id]["specificity"]:.2f}          '
          f' {metrics_per_class[class_id]["f1_score"]:.2f}          '
          f' {metrics_per_class[class_id]["accuracy"]:.2f}'
      )

  print(
      '\nclass id          Sensitivity(Recall)    Specificity    '
      ' severe_missed_df          ppv          npv'
  )
  for class_map_tag in metrics_per_binary_class.keys():
    print(
        f'\n{class_map_tag}         '
        f' {metrics_per_binary_class[class_map_tag]["sensitivity"]:.2f}        '
        f'  {metrics_per_binary_class[class_map_tag]["specificity"]:.2f}       '
        f'   {metrics_per_binary_class[class_map_tag]["pc_severe_missed"]:.2f}%'
        f'            {metrics_per_binary_class[class_map_tag]["ppv"]:.2f}     '
        f'     {metrics_per_binary_class[class_map_tag]["npv"]:.2f}'
    )


def evaluate_by_gender(
    eval_df: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_columns: List[str],
) -> None:
  """Evaluates regression metrics by gender.

  Args:
    eval_df: Evaluation results dataframe consisting of following columns: 
      - uuid: Patient/sample identifier.
      - gt_hgb: Ground truth hemoglobin value in g/dL.
      - gt_anemia_label: Ground truth anemia label.
      - pred_hgb: Predicted hemoglobin value in g/dL.
      - pred_anemia_label: Predicted anemia label.
    metadata:  Dataset Metadata DataFrame indexed by patient/sample UUIDs.
    feature_columns: Feature columns to select from metadata.
      Expected: [ 'Age', 'ground_truth_hgb', 'ground_truth_anemia_label',
        'PregnancyStatus', 'Gender', ]
      Note: Feature columns should be a subset of the metadata columns.

  Returns:
    None
  """

  gender_metadata = metadata[metadata.Gender == 'Female']
  if len(gender_metadata) > 0:
    print('\n Results for Gender==Female : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, gender_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )

  gender_metadata = metadata[metadata.Gender == 'Male']
  if len(gender_metadata) > 0:
    print('\n Results for Gender==Male : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, gender_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )

  gender_metadata = metadata[metadata.Gender == 'Other']
  if len(gender_metadata) > 0:
    print("\n Results for Gender==Other : \n")
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, gender_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )


def evaluate_by_pregnancy_status(
    eval_df: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_columns: List[str],
) -> None:
  """Evaluates regression metrics by pregnancy status.

  Args:
    eval_df: Evaluation results dataframe consisting of following columns:
      - uuid: Patient/sample identifier.
      - gt_hgb: Ground truth hemoglobin value in g/dL.
      - gt_anemia_label: Ground truth anemia label.
      - pred_hgb: Predicted hemoglobin value in g/dL.
      - pred_anemia_label: Predicted anemia label.
    metadata:  Dataset Metadata DataFrame indexed by patient/sample UUIDs.
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

  Returns:
    None
  """

  pregnant_metadata = metadata[metadata.Gender == 'Female']
  pregnant_metadata = pregnant_metadata[
      pregnant_metadata.PregnancyStatus == 'PREGNANT']
  if len(pregnant_metadata) > 0:
    print('\n Results for PregnancyStatus == PREGNANT : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, pregnant_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )

  pregnant_metadata = metadata[metadata.Gender == 'Female']
  pregnant_metadata = pregnant_metadata[
      pregnant_metadata.PregnancyStatus == 'NOT PREGNANT']
  if len(pregnant_metadata) > 0:
    print('\n Results for PregnancyStatus == NOT PREGNANT : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, pregnant_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )


def evaluate_by_age(
    eval_df: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_columns: List[str],
) -> None:
  """Evaluates regression metrics by age.

  Args:
    eval_df: Evaluation results dataframe consisting of following columns:
      -uuid: Patient/sample identifier.
      - gt_hgb: Ground truth hemoglobin value in g/dL.
      - gt_anemia_label: Ground truth anemia label.
      - pred_hgb: Predicted hemoglobin value in g/dL.
      - pred_anemia_label: Predicted anemia label.
    metadata:  Dataset Metadata DataFrame indexed by patient/sample UUIDs.
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

  Returns:
    None
  """

  age_metadata = metadata[
      (metadata.AgeNumeric >= 0) & (metadata.AgeNumeric <= 19)
  ]
  age_metadata = age_metadata[
      (age_metadata.Gender == 'Male') | (age_metadata.Gender == 'Female')]
  if len(age_metadata) > 0:
    print('\n Results for Age>=0 & Age <= 19 : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, age_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )

  age_metadata = metadata[metadata.AgeNumeric > 19]
  age_metadata = age_metadata[
      (age_metadata.Gender == 'Male') | (age_metadata.Gender == 'Female')
  ]
  if len(age_metadata) > 0:
    print('\n Results for Age > 19 : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, age_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )


def evaluate_by_sickle_cell_status(
    eval_df: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_columns: List[str],
) -> None:
  """Evaluates regression metrics by sickle cell status.

  Args:
    eval_df: Evaluation results dataframe consisting of following columns:
      - uuid: Patient/sample identifier.
      - gt_hgb: Ground truth hemoglobin value in g/dL.
      - gt_anemia_label: Ground truth anemia label.
      - pred_hgb: Predicted hemoglobin value in g/dL.
      - pred_anemia_label: Predicted anemia label.
    metadata:  Dataset Metadata DataFrame indexed by patient/sample UUIDs.
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

  Returns:
    None
  """

  sickle_metadata = metadata[metadata.SickleCellDisease == 'DISEASE']
  if len(sickle_metadata) > 0:
    print('\n Results for Sickle==DISEASE : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, sickle_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )

  sickle_metadata = metadata[metadata.SickleCellDisease == 'NEGATIVE']
  if len(sickle_metadata) > 0:
    print('\n Results for Sickle==NEGATIVE : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, sickle_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )

  sickle_metadata = metadata[metadata.SickleCellDisease == 'NOT_KNOWN']
  if len(sickle_metadata) > 0:
    print('\n Results for Sickle==NOT_KNOWN : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, sickle_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )

  sickle_metadata = metadata[metadata.SickleCellDisease == 'TRAIT']
  if len(sickle_metadata) > 0:
    print('\n Results for Sickle==TRAIT : \n')
    metrics_per_class, metrics_per_binary_class = sub_group_metrics_regression(
        eval_df, sickle_metadata, feature_columns
    )
    print_metrics(metrics_per_class, metrics_per_binary_class)
    print(
        '\n -------------------------------------------------------------------------------------------------------- \n'
    )
