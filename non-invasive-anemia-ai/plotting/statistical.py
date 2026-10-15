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
"""Utility functions for statistical plotting."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color scheme definitions for anemia severity categories
colors = {
    'normal': 'lightgreen',
    'mild': 'skyblue',
    'moderate': 'orange',
    'severe': 'tomato',
}


def plot_predicted_vs_actual_hgb(
    eval_df: pd.DataFrame,
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
  """Plots Predicted vs Actual Hemoglobin (Hgb) scatter plot.

  Args:
    eval_df: Evaluation DataFrame containing `gt_hgb`, `pred_hgb`, and
      `pred_anemia_label`.
    show_plot: Whether to invoke plt.show().
    save_path: Optional file path to save the plot figure.

  Returns:
    The generated matplotlib Figure object.
  """
  fig, ax = plt.subplots(figsize=(8, 6))

  for label, color in colors.items():
    if 'pred_anemia_label' in eval_df.columns:
      mask = eval_df['pred_anemia_label'] == label
    else:
      mask = pd.Series([True] * len(eval_df))
    if mask.any():
      ax.scatter(
          eval_df.loc[mask, 'gt_hgb'],
          eval_df.loc[mask, 'pred_hgb'],
          c=color,
          label=label,
          alpha=0.6,
          edgecolors='none',
      )

  all_values = np.concatenate(
      (eval_df['gt_hgb'].dropna().values, eval_df['pred_hgb'].dropna().values)
  )
  if len(all_values) > 0:
    min_val, max_val = float(np.min(all_values)), float(np.max(all_values))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

  ax.set_xlabel('Actual HgB (g/dL)')
  ax.set_ylabel('Predicted HgB (g/dL)')
  ax.set_title(f'Predicted vs Actual HgB (n={len(eval_df)})')
  ax.legend()

  if save_path:
    fig.savefig(save_path, bbox_inches='tight')
  if show_plot:
    plt.show()

  return fig


def plot_bland_altman(
    eval_df: pd.DataFrame,
    show_plot: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
  """Generates a Bland-Altman agreement plot for regression predictions.

  Args:
    eval_df: DataFrame containing `gt_hgb`, `pred_hgb`, and optional
      `pred_anemia_label`.
    show_plot: Whether to invoke plt.show().
    save_path: Optional path to save the figure image.

  Returns:
    The generated matplotlib Figure object.
  """
  df = eval_df.copy()
  df['mean_actual_predicted_hb'] = (df['gt_hgb'] + df['pred_hgb']) / 2.0
  df['error'] = df['gt_hgb'] - df['pred_hgb']

  md = float(df['error'].mean())
  sd = float(df['error'].std())
  upper_loa = md + 1.96 * sd
  lower_loa = md - 1.96 * sd

  fig, ax = plt.subplots(figsize=(10, 6))

  for label, color in colors.items():
    if 'pred_anemia_label' in df.columns:
      mask = df['pred_anemia_label'] == label
    else:
      mask = pd.Series([True] * len(df))
    if mask.any():
      ax.scatter(
          df.loc[mask, 'mean_actual_predicted_hb'],
          df.loc[mask, 'error'],
          c=color,
          label=label,
          alpha=0.7,
          edgecolors='none',
      )

  # Statistical Reference Lines
  ax.axhline(md, color='red', linestyle='--', label=f'Mean Diff: {md:.2f}')
  ax.axhline(
      upper_loa, color='gray', linestyle=':', label=f'+1.96 SD: {upper_loa:.2f}'
  )
  ax.axhline(
      lower_loa, color='gray', linestyle=':', label=f'-1.96 SD: {lower_loa:.2f}'
  )

  ax.set_xlabel('Mean of Actual & Predicted Hgb (g/dL)')
  ax.set_ylabel('Difference (Actual - Predicted) Hgb')
  ax.set_title('Bland-Altman Plot [Regression]')
  ax.legend()
  fig.tight_layout()

  if save_path:
    fig.savefig(save_path, bbox_inches='tight')
  if show_plot:
    plt.show()

  return fig
