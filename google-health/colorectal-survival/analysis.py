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
"""Statistical analysis."""

import lifelines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

TIME = 'time'
OBSERVED = 'observed'
RISK_SCORE = 'risk_score'


def plot_km_curve(df_tune, df_test):
  """Returns KM curves for each risk group for `df_test`.

  Risk groups are defined via thresholds computed on `df_tune`.

  Args:
    df_tune: a pd.DataFrame of tune set data.
    df_test: a pd.DataFrame of test set data.
  """
  # Compute risk groups
  df_test['risk_group'] = discretize(df_tune[RISK_SCORE], df_test[RISK_SCORE])

  # Plot KM curves per risk group
  fig, ax = plt.subplots()
  groups = ['Low Risk', 'Medium Risk', 'High Risk']
  kmfs = []
  for group in groups:
    kmf = lifelines.KaplanMeierFitter()
    df_group = df_test.query(f"risk_group=='{group}'")
    if df_group.empty:
      continue
    kmf.fit(df_group[TIME], event_observed=df_group[OBSERVED], label=group)
    kmf.plot(ax=ax)
    kmfs.append(kmf)
  lifelines.plotting.add_at_risk_counts(*kmfs, ax=ax)
  return fig


def discretize(risk_scores_tune, risk_scores_test):
  """Discretize `risk_scores_test` based on thresholds from `risk_scores_tune`.

  Args:
    risk_scores_tune: np.ndarray of continuous risk scores.
    risk_scores_test: np.ndarray of continuous risk scores.

  Returns:
    an np.ndarray of disretized test set risk scores.
  """
  thresholds_valid = np.percentile(risk_scores_tune, [25, 75])
  risk_groups_test = np.digitize(risk_scores_test, bins=thresholds_valid)
  risk_groups_test = pd.Series(risk_groups_test)
  risk_group_map = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
  risk_groups_test = risk_groups_test.apply(lambda x: risk_group_map[x])
  return risk_groups_test


def c_index(df):
  return lifelines.utils.concordance_index(df[TIME], df[RISK_SCORE],
                                           df[OBSERVED])


def survival_auc(df, threshold):
  """Survival AUC."""
  df_binarized = binarize_time(df, threshold)
  return sklearn.metrics.roc_auc_score(df_binarized[TIME],
                                       df_binarized[RISK_SCORE])


def binarize_time(df, threshold):
  """Binarize time based on threshold.

  If time > threshold: `observed` and `time` columns are set to 1.
  If time <= threshold: unobserved examples are dropped and `time` is set to 0.

  Args:
    df: pd.DataFrame containing `time` and `observed` columns.
    threshold: the time threshold on which to binarize.

  Returns:
    a pd.Dataframe where time has been discretized.
  """

  def update_observed(row):
    if row[TIME] > threshold:
      return 1
    return row[OBSERVED]

  df[OBSERVED] = df.apply(update_observed, axis=1)
  df[TIME] = (df[TIME] > threshold).astype(int)

  # Remove censored examples below threshold. These examples cannot be
  # compared to any others.
  df = df.query('time != 0 or observed != 0')
  return df


def get_hazard_ratios(df_test):
  cph = lifelines.CoxPHFitter()
  cph.fit(df_test, duration_col=TIME, event_col=OBSERVED)
  return cph.summary
